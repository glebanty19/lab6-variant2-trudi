#!/usr/bin/env python3
"""Faster R-CNN experiments on TRUDI's official aerial COCO split."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from detection_metrics import average_precision, best_threshold, collate, iou

DATA = Path(__file__).resolve().parents[1] / "datasets" / "trudi" / "coco" / "aerial"
WASTE_CKPT = Path(__file__).resolve().parent / "runs" / "dronewaste_piles_fine" / "best.pt"
SPLITS = {"train": "train", "val": "val", "test": "test"}


def read_split(split: str):
    folder = DATA / SPLITS[split]
    coco = json.loads((folder / "annotations.json").read_text(encoding="utf-8"))
    # The assignment asks for container detection. Treat tank containers as containers;
    # exclude trailers so the target class has a precise meaning.
    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    image_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_id: dict[int, list[dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        if id_to_name[ann["category_id"]] in ("container", "tank_container"):
            anns_by_id.setdefault(ann["image_id"], []).append(ann)
    return folder, coco, image_by_id, anns_by_id


def degrade_image(image: Image.Image, factor: float) -> Image.Image:
    """Simulate coarser spatial sampling while retaining original box coordinates."""
    if factor <= 1:
        return image
    width, height = image.size
    return image.resize((max(1, round(width/factor)), max(1, round(height/factor))),
                        Image.Resampling.BOX).resize((width, height), Image.Resampling.BILINEAR)


class TrudiDataset:
    def __init__(self, split: str, training: bool = False, degrade: int = 1,
                 scale_aug: bool = False):
        self.split = split
        self.folder, self.coco, self.images, self.anns = read_split(split)
        self.records = self.coco["images"]
        self.training = training
        self.degrade = degrade
        self.scale_aug = scale_aug

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch
        rec = self.records[index]
        with Image.open(self.folder / rec["file_name"]) as source:
            image = source.convert("RGB")
        factor = random.choice((1, 2, 4)) if self.scale_aug else self.degrade
        image = degrade_image(image, factor)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1)
        anns = self.anns.get(rec["id"], [])
        boxes = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([max(0, x), max(0, y), min(rec["width"], x+w), min(rec["height"], y+h)])
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target = {
            "boxes": boxes,
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "image_id": torch.tensor([rec["id"]], dtype=torch.int64),
            "area": (boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        if self.training and random.random() < 0.5:
            tensor = tensor.flip(-1)
            if len(boxes):
                x0, x1 = boxes[:, 0].clone(), boxes[:, 2].clone()
                boxes[:, 0] = rec["width"] - x1
                boxes[:, 2] = rec["width"] - x0
        return tensor, target


def make_model(torchvision, FastRCNNPredictor, source_ckpt: Path | None = None,
               resume: Path | None = None, min_size=640, max_size=1024):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.transform.min_size = (min_size,)
    model.transform.max_size = max_size
    model.roi_heads.detections_per_img = 250
    if resume:
        state = torch_load(resume)["model"]
        model.load_state_dict(state)
    elif source_ckpt:
        state = torch_load(source_ckpt)["model"]
        own = model.state_dict()
        transferable = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        missing, unexpected = model.load_state_dict(transferable, strict=False)
        print(f"Warm start: loaded {len(transferable)} compatible tensors; "
              f"new task head tensors={len(missing)}", flush=True)
    return model


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def evaluate(model, split: str, degrade: float = 1, tile: int | None = None,
             overlap: float = 0.2, context: int = 0, device=None):
    import torch
    from torchvision.ops import nms
    folder, coco, image_by_id, anns_by_id = read_split(split)
    preds, truths = [], {}
    model.eval()
    for rec in coco["images"]:
        with Image.open(folder / rec["file_name"]) as source:
            image = source.convert("RGB")
        image = degrade_image(image, degrade)
        w, h = image.size
        if tile:
            step = max(1, int(tile * (1-overlap)))
            xs = list(range(0, max(1,w-tile+1), step)); ys = list(range(0,max(1,h-tile+1),step))
            if not xs or xs[-1] != max(0,w-tile): xs.append(max(0,w-tile))
            if not ys or ys[-1] != max(0,h-tile): ys.append(max(0,h-tile))
            windows = [(x,y,min(w,x+tile),min(h,y+tile)) for y in ys for x in xs]
        else:
            windows = [(0,0,w,h)]
        boxes, scores = [], []
        for x0,y0,x1,y1 in windows:
            cx0,cy0,cx1,cy1=max(0,x0-context),max(0,y0-context),min(w,x1+context),min(h,y1+context)
            crop = image.crop((cx0,cy0,cx1,cy1))
            arr = np.asarray(crop,dtype=np.float32).copy()/255.0
            tensor = torch.from_numpy(arr).permute(2,0,1).to(device)
            with torch.no_grad(): pred = model([tensor])[0]
            for b,s,l in zip(pred["boxes"].cpu().tolist(),pred["scores"].cpu().tolist(),pred["labels"].cpu().tolist()):
                if int(l) == 1:
                    global_box=[b[0]+cx0,b[1]+cy0,b[2]+cx0,b[3]+cy0]
                    center_x=(global_box[0]+global_box[2])/2
                    center_y=(global_box[1]+global_box[3])/2
                    if x0 <= center_x < x1 and y0 <= center_y < y1:
                        boxes.append(global_box); scores.append(float(s))
        if boxes:
            keep = nms(torch.tensor(boxes,device=device),torch.tensor(scores,device=device),0.5).cpu().tolist()
            boxes=[boxes[i] for i in keep]; scores=[scores[i] for i in keep]
        iid = str(rec["id"])
        preds.append({"image_id":iid,"boxes":boxes,"labels":[1]*len(boxes),"scores":scores})
        truth=[]
        id_to_name={c["id"]:c["name"] for c in coco["categories"]}
        for ann in anns_by_id.get(rec["id"],[]):
            x,y,bw,bh=ann["bbox"]
            truth.append({"box":[x,y,x+bw,y+bh],"label":1})
        truths[iid]=truth
    ap_values=[]
    for iou_thr in np.arange(.5,.951,.05):
        ap,_,_,support=average_precision(preds,truths,1,float(iou_thr))
        if math.isfinite(ap): ap_values.append(ap)
    ap50,recalls,precisions,support=average_precision(preds,truths,1,.5)
    return {"split":split,"degrade_factor":degrade,"tile_size":tile,"overlap":overlap if tile else None,
            "context_px":context if tile else 0,
            "images":len(coco["images"]),"support":support,"AP50":ap50,
            "mAP50_95":float(np.mean(ap_values)),"recall_curve":recalls,
            "precision_curve":precisions},preds,truths


def train(args):
    import torch, torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    train_ds=TrudiDataset("train",True,args.degrade_train,args.scale_aug)
    loader=torch.utils.data.DataLoader(train_ds,batch_size=1,shuffle=True,num_workers=args.workers,collate_fn=collate,pin_memory=device.type=="cuda")
    model=make_model(torchvision,FastRCNNPredictor,
                     source_ckpt=args.source_checkpoint if not args.resume else None,
                     resume=args.resume,min_size=args.min_size,max_size=args.max_size).to(device)
    if args.freeze_backbone:
        for p in model.parameters(): p.requires_grad=False
        for p in model.roi_heads.box_predictor.parameters():p.requires_grad=True
    optim=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=args.lr,momentum=.9,weight_decay=.0005)
    sched=torch.optim.lr_scheduler.StepLR(optim,max(1,args.epochs//3),gamma=.1)
    best=-1; history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for k,(images,targets) in enumerate(loader,1):
            images=[x.to(device,non_blocking=True) for x in images]
            targets=[{key:v.to(device,non_blocking=True) for key,v in t.items()} for t in targets]
            loss=sum(model(images,targets).values())
            optim.zero_grad(set_to_none=True); loss.backward(); optim.step(); losses.append(float(loss.detach()))
            if k%50==0: print(json.dumps({"epoch":epoch,"batch":k,"loss":float(np.mean(losses[-50:]))}),flush=True)
        sched.step()
        vm,_,_=evaluate(model,"val",args.degrade_train if args.degrade_train>1 else 1,device=device)
        row={"epoch":epoch,"train_loss":float(np.mean(losses)),"val_map50":vm["AP50"],"val_map50_95":vm["mAP50_95"]}
        history.append(row); print(json.dumps(row),flush=True)
        ck={"model":model.state_dict(),"epoch":epoch,"classes":{"1":"container + tank_container"},
            "min_size":args.min_size,"max_size":args.max_size,"seed":args.seed,"validation":row,
            "data_root":str(DATA),"train_images":len(train_ds),"source_checkpoint":str(args.source_checkpoint) if args.source_checkpoint and not args.resume else None,
            "resume_checkpoint":str(args.resume) if args.resume else None,"torch":torch.__version__,"torchvision":torchvision.__version__}
        torch.save(ck,out/"last.pt")
        if row["val_map50_95"]>best: best=row["val_map50_95"];torch.save(ck,out/"best.pt")
        (out/"history.json").write_text(json.dumps(history,indent=2),encoding="utf8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(9,3.5)); epochs=[x["epoch"] for x in history]
    ax[0].plot(epochs,[x["train_loss"] for x in history],marker="o");ax[0].set_title("Train loss")
    ax[1].plot(epochs,[x["val_map50_95"] for x in history],marker="o");ax[1].set_title("Validation mAP@[.5:.95]")
    for a in ax:a.set_xlabel("Epoch");a.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/"training_curves.png",dpi=160);plt.close(fig)
    complete={"epochs":len(history),"best_epoch":max(history,key=lambda row:row["val_map50_95"])["epoch"],
              "best_val_map50_95":max(row["val_map50_95"] for row in history),
              "source_checkpoint":str(args.source_checkpoint) if not args.resume else None,
              "resume_checkpoint":str(args.resume) if args.resume else None}
    (out/"training_complete.json").write_text(json.dumps(complete,indent=2),encoding="utf8")


def eval_command(args):
    import torch,torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    model=make_model(torchvision,FastRCNNPredictor,resume=args.checkpoint,min_size=ck.get("min_size",640),max_size=ck.get("max_size",1024))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model.to(device)
    result,preds,truths=evaluate(model,args.split,args.degrade,args.tile,args.overlap,args.context,device)
    if args.split=="val":
        threshold,point=best_threshold(preds,truths,[1]);result["best_f1_threshold"]={"confidence":threshold,**point}
    elif args.threshold is not None:
        tp=fp=fn=0
        for pred in preds:
            matched=set()
            ranked=sorted([(s,b) for b,s in zip(pred["boxes"],pred["scores"]) if s>=args.threshold],reverse=True)
            for _,box in ranked:
                choices=[(iou(box,t["box"]),j) for j,t in enumerate(truths[pred["image_id"]]) if j not in matched]
                overlap,index=max(choices,default=(0.0,-1))
                if overlap>=.5:tp+=1;matched.add(index)
                else:fp+=1
            fn+=len(truths[pred["image_id"]])-len(matched)
        precision=tp/(tp+fp) if tp+fp else 0
        recall=tp/(tp+fn) if tp+fn else 0
        result["threshold"]=args.threshold
        result["operating_point"]={"tp":tp,"fp":fp,"fn":fn,"precision":precision,"recall":recall,"f1":2*precision*recall/(precision+recall) if precision+recall else 0}
    clean={k:v for k,v in result.items() if k not in ("recall_curve","precision_curve")}
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    (out/(args.prefix+"_metrics.json")).write_text(json.dumps(clean,indent=2),encoding="utf8")
    import gzip
    with gzip.open(out/(args.prefix+"_detections.json.gz"),"wt",encoding="utf8") as stream:
        json.dump({"predictions":preds,"truths":truths},stream)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,4));plt.plot(result["recall_curve"],result["precision_curve"]);plt.xlabel("Recall");plt.ylabel("Precision");plt.grid(alpha=.25);plt.tight_layout();plt.savefig(out/(args.prefix+"_pr.png"),dpi=160);plt.close()
    print(json.dumps(clean,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="command",required=True)
    t=sub.add_parser("train");t.add_argument("--output",type=Path,required=True);t.add_argument("--epochs",type=int,default=6);t.add_argument("--seed",type=int,default=42);t.add_argument("--lr",type=float,default=.001);t.add_argument("--min-size",type=int,default=640);t.add_argument("--max-size",type=int,default=1024);t.add_argument("--workers",type=int,default=2);t.add_argument("--degrade-train",type=int,choices=(1,2,4),default=1);t.add_argument("--scale-aug",action="store_true");t.add_argument("--freeze-backbone",action="store_true");t.add_argument("--source-checkpoint",type=Path,default=WASTE_CKPT);t.add_argument("--resume",type=Path,default=None);t.set_defaults(func=train)
    e=sub.add_parser("evaluate");e.add_argument("--checkpoint",type=Path,required=True);e.add_argument("--output",type=Path,required=True);e.add_argument("--split",choices=("val","test"),default="test");e.add_argument("--prefix",default="eval");e.add_argument("--degrade",type=float,choices=(1,1.5,2,3,4),default=1);e.add_argument("--tile",type=int,default=None);e.add_argument("--overlap",type=float,default=.2);e.add_argument("--context",type=int,default=0);e.add_argument("--threshold",type=float,default=None);e.set_defaults(func=eval_command)
    args=parser.parse_args();args.func(args)


if __name__=="__main__":main()
