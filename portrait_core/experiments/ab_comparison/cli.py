from __future__ import annotations
import argparse,json
from datetime import datetime
from pathlib import Path
from .comparison import ComparisonValidationError,compare_datasets
from .report import generate_pdf

def main(argv=None):
    ap=argparse.ArgumentParser(description="Compare two ORION Dataset Archives")
    ap.add_argument("--dataset-a",required=True);ap.add_argument("--dataset-b",required=True)
    ap.add_argument("--label-a",default="fixed_step");ap.add_argument("--label-b",default="quality_profile")
    ap.add_argument("--output");ap.add_argument("--bootstrap",type=int,default=0,help="reserved; written to provenance")
    ap.add_argument("--force",action="store_true");args=ap.parse_args(argv)
    root=Path(args.output) if args.output else Path(args.dataset_a).resolve().parent/"experiments"/("AB-"+datetime.now().strftime("%Y%m%d-%H%M%S"))
    if root.exists() and any(root.iterdir()) and not args.force: ap.error(f"output is not empty: {root}")
    root.mkdir(parents=True,exist_ok=True)
    try: result=compare_datasets(args.dataset_a,args.dataset_b,label_a=args.label_a,label_b=args.label_b,force=args.force)
    except ComparisonValidationError as exc: ap.error(str(exc))
    result["provenance"]["bootstrap_iterations"]=args.bootstrap
    jp=root/"ab_comparison.json";jp.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    pp=generate_pdf(result,root/"ab_comparison.pdf");print(json.dumps({"status":result['control_validation']['status'],"json":str(jp),"pdf":str(pp)},ensure_ascii=False));return 0
