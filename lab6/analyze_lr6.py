"""Measure LR6 image statistics and ROI feature shift."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from trudi_lr6 import DATA as TRUDI, degrade_image, make_model
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"runs/variant2_audit"

def lr6_shift(checkpoint: Path, sample: int) -> None:
    import torch, torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    coco=json.loads((TRUDI/"test/annotations.json").read_text(encoding="utf8"))
    records=coco["images"][:sample]
    checkpoint_data=torch.load(checkpoint,map_location="cpu",weights_only=False)
    model=make_model(torchvision,FastRCNNPredictor,resume=checkpoint,
                     min_size=checkpoint_data.get("min_size",640),max_size=checkpoint_data.get("max_size",1024))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    current=[]
    def hook(module, inputs, output):
        current.append(output.detach().float().mean(dim=0).cpu().numpy())
    handle=model.roi_heads.box_head.fc7.register_forward_hook(hook)
    hist={}; features={}
    for factor in (1,2,4):
        channels=[];vectors=[]
        for rec in records:
            with Image.open(TRUDI/"test"/rec["file_name"]) as source:
                image=degrade_image(source.convert("RGB"),factor)
            small=image.resize((256,256),Image.Resampling.BOX)
            channels.append(np.asarray(small,dtype=np.float32).reshape(-1,3)/255.)
            arr=np.asarray(image,dtype=np.float32).copy()/255.
            current.clear()
            with torch.inference_mode():model([torch.from_numpy(arr).permute(2,0,1).to(device)])
            if current: vectors.append(current[-1])
        pixels=np.concatenate(channels)
        hist[str(factor)]={"mean":pixels.mean(axis=0).tolist(),"std":pixels.std(axis=0).tolist(),
                           "bins":[np.histogram(pixels[:,c],bins=32,range=(0,1),density=True)[0].tolist() for c in range(3)]}
        features[str(factor)]=np.mean(vectors,axis=0)
    handle.remove()
    base=features["1"]
    distances={factor:{"l2":float(np.linalg.norm(vector-base)),
                       "relative_l2":float(np.linalg.norm(vector-base)/(np.linalg.norm(base)+1e-12))}
               for factor,vector in features.items()}
    output={"images":len(records),"split":"test","feature":"Faster R-CNN ROI box head fc7, mean of proposals per image then mean of images",
            "channel_statistics":hist,"feature_mean_distance":distances}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"lr6_feature_shift.json").write_text(json.dumps(output,indent=2),encoding="utf8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11,3.5))
    centers=np.linspace(1/64,1-1/64,32)
    for c,color in enumerate(("red","green","blue")):
        for factor,style in (("1","-"),("2","--"),("4",":")):
            axes[c].plot(centers,hist[factor]["bins"][c],color=color,linestyle=style,label="x"+factor)
        axes[c].set(xlabel="Normalized brightness",ylabel="Density",title=("R","G","B")[c]);axes[c].legend()
    fig.tight_layout();fig.savefig(OUT/"lr6_rgb_histograms.png",dpi=160);plt.close(fig)
    print(json.dumps({"images":len(records),"distances":distances},indent=2))



if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--sample",type=int,default=20)
    args=parser.parse_args()
    lr6_shift(args.checkpoint,args.sample)
