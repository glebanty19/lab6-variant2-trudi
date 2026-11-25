#!/usr/bin/env python3
"""Reproduce LR6 variant 2 using the completed LR4 backbone and TRUDI labels."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE="trudi_lr4_source"
CONTROL="trudi_lr4_continue_native"
AUG="trudi_lr4_scale_aug"
ADAPT="trudi_lr4_adapt_x4"


def run(*arguments: str) -> None:
    print("RUN", " ".join(arguments),flush=True)
    subprocess.run([sys.executable,*arguments],cwd=ROOT,check=True)


def selected_lr4_checkpoint() -> str:
    choice=ROOT/"runs/variant2_audit/selected_lr4_model.json"
    if choice.is_file():
        return json.loads(choice.read_text(encoding="utf8"))["checkpoint"]
    return "runs/dronewaste_piles_fine/best.pt"


def train(name: str,epochs: int,extra: tuple[str,...],force: bool) -> None:
    completed=ROOT/"runs"/name/"training_complete.json"
    if completed.is_file() and not force:
        return
    run("trudi_lr6.py","train","--output",f"runs/{name}","--epochs",str(epochs),
        "--seed","42","--lr","0.001","--workers","2",*extra)


def evaluate(name: str,mode: str,split: str="test",extra: tuple[str,...]=(),force: bool=False) -> None:
    result=ROOT/"runs"/name/mode/f"{mode}_metrics.json"
    if result.is_file() and not force:
        return
    threshold=None
    if split=="test":
        val_mode="val_x4" if name==ADAPT else "val"
        val=ROOT/"runs"/name/val_mode/f"{val_mode}_metrics.json"
        threshold=json.loads(val.read_text(encoding="utf8"))["best_f1_threshold"]["confidence"]
    cmd=["trudi_lr6.py","evaluate","--checkpoint",f"runs/{name}/best.pt",
         "--output",f"runs/{name}/{mode}","--split",split,"--prefix",mode]
    if threshold is not None:
        cmd.extend(("--threshold",str(threshold)))
    run(*cmd,*extra)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force",action="store_true",help="Retrain and reevaluate existing LR6 checkpoints")
    args=parser.parse_args()
    selection=ROOT/"runs/variant2_audit/selected_lr4_model.json"
    if not selection.is_file():
        raise FileNotFoundError(f"Copy the selected LR4 model manifest to {selection}")
    source=selected_lr4_checkpoint()
    if not (ROOT/source).is_file():
        raise FileNotFoundError(ROOT/source)
    run("audit_lr6.py")
    train(SOURCE,5,("--source-checkpoint",source),args.force)
    train(CONTROL,4,("--resume",f"runs/{SOURCE}/best.pt"),args.force)
    train(AUG,4,("--resume",f"runs/{SOURCE}/best.pt","--scale-aug"),args.force)
    train(ADAPT,3,("--resume",f"runs/{SOURCE}/best.pt","--degrade-train","4"),args.force)
    evaluate(SOURCE,"val",split="val",force=args.force)
    evaluate(CONTROL,"val",split="val",force=args.force)
    evaluate(AUG,"val",split="val",force=args.force)
    evaluate(ADAPT,"val_x4",split="val",extra=("--degrade","4"),force=args.force)
    for name in (SOURCE,CONTROL,AUG):
        for mode,factor in (("test_native","1"),("test_x1_5","1.5"),("test_x2","2"),
                            ("test_x3","3"),("test_x4","4")):
            evaluate(name,mode,extra=("--degrade",factor),force=args.force)
    for mode,factor in (("test_native","1"),("test_x2","2"),("test_x4","4")):
        evaluate(ADAPT,mode,extra=("--degrade",factor),force=args.force)
    evaluate(SOURCE,"test_tiles",extra=("--tile","640","--overlap","0.2"),force=args.force)
    evaluate(SOURCE,"test_tiles_context",extra=("--tile","640","--overlap","0.2","--context","128"),force=args.force)
    evaluate(SOURCE,"test_tiles_800_context",extra=("--tile","800","--overlap","0.2","--context","128"),force=args.force)
    run("analyze_lr6.py","--checkpoint",f"runs/{SOURCE}/best.pt","--sample","20")
    run("build_trudi_lr6_report.py")
    print("LR6 variant 2 complete:",ROOT.parent/"Звіт_Лаб6_Варіант2.docx",flush=True)


if __name__=="__main__":
    main()
