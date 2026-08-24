"""Scientific A/B comparison of two ORION Dataset Archives."""
from __future__ import annotations

import json, math, platform, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from portrait_core.invariants.ratio_engine import build_invariant_set_from_pfr
from portrait_core.invariants.registry import INVARIANT_DEFINITIONS
from portrait_core.lic_stability_report import build_lic_stability_report
from .statistics import describe, relative_change

ISSUES = ("head_yaw", "head_pitch", "head_roll", "frame_blur", "brightness", "contrast",
          "face_too_small", "face_too_large", "face_out_of_frame")

class ComparisonValidationError(ValueError): pass

def _read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))

def _num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)): return None
    return float(v) if math.isfinite(float(v)) else None

def _flat(v, prefix=""):
    out = {}
    if isinstance(v, dict):
        for k, x in v.items(): out.update(_flat(x, f"{prefix}.{k}" if prefix else str(k)))
    elif _num(v) is not None and prefix: out[prefix] = float(v)
    return out

def _archive(directory):
    root = Path(directory).resolve(); manifest = _read(root / "dataset.json")
    records = []
    for item in manifest.get("items", []):
        rel = item.get("pfr_path")
        path = root / rel if rel else None
        records.append({"item": item, "pfr": _read(path) if path and path.is_file() else None})
    return {"root": root, "manifest": manifest, "records": records}

def _identity(d):
    m=d["manifest"]; s=m.get("settings",{}); ab=m.get("analysis_backend",{}); model=ab.get("model",{})
    p=next((r["pfr"] for r in d["records"] if r["pfr"]),{}); video=m.get("video_source",{})
    return {"dataset_id":m.get("id"), "source":m.get("source"), "backend":ab.get("name") or s.get("backend"),
            "model_id":model.get("id") or model.get("model_id") or s.get("model_path"),
            "model_sha256":model.get("sha256"), "generator":p.get("generator") or p.get("metadata",{}).get("generator"),
            "video":{k:video.get(k) for k in ("width","height","fps","codec","format_id","adapter")}}

def _validate(a,b):
    ai,bi=_identity(a),_identity(b); critical={"source","backend","model_id","model_sha256"}; checks=[]
    for key in ("source","backend","model_id","model_sha256","generator","video"):
        checks.append({"field":key,"a":ai.get(key),"b":bi.get(key),"match":ai.get(key)==bi.get(key),
                       "severity":"critical" if key in critical else "warning"})
    failed=[x for x in checks if not x["match"] and x["severity"]=="critical"]
    warns=[x for x in checks if not x["match"] and x["severity"]=="warning"]
    return {"status":"invalid_comparison" if failed else ("partially_controlled" if warns else "controlled"),
            "checks":checks,"critical_mismatches":[x["field"] for x in failed],"warnings":[x["field"] for x in warns],"a":ai,"b":bi}

def _selection(d,label):
    s=d["manifest"].get("settings",{}); keys=("frame_selection_mode","selection_profile","frame_step",
      "target_selected_frames","min_temporal_distance_seconds","max_frames_per_episode","max_abs_yaw_deg",
      "max_abs_pitch_deg","max_abs_roll_deg","require_closed_mouth","require_open_eyes","use_gaze_score")
    out={k:s.get(k) for k in keys}; out["missing_provenance_fields"]=[k for k in keys if k not in s]
    if out["frame_selection_mode"] is None and label=="A": out.update(frame_selection_mode="fixed_step",mode_inferred=True)
    return out

def _records(d,status=None): return [r for r in d["records"] if r["pfr"] and (status is None or r["item"].get("status")==status)]

def _quality(d):
    items=[r["item"] for r in d["records"]]; n=len(items); statuses=Counter(str(x.get("status") or "unknown") for x in items)
    issue=Counter(c for x in items for c in x.get("issue_codes",[])); codes=sorted(set(ISSUES)|set(issue))
    return {"total":n,"pfr_count":sum(r["pfr"] is not None for r in d["records"]),"statuses":dict(statuses),
      "completion_rate":sum(r["pfr"] is not None for r in d["records"])/n if n else 0,
      "issue_codes":{c:{"count":issue[c],"rate":issue[c]/n if n else 0} for c in codes}}

def _stats(records, invariants=False):
    values=defaultdict(list)
    for r in records:
        if invariants:
            for name,ratio in build_invariant_set_from_pfr(r["pfr"]).ratios.items():
                if ratio.valid and ratio.value is not None: values[name].append(float(ratio.value))
        else:
            for name,value in _flat(r["pfr"].get("measurements",{})).items(): values[name].append(value)
    names=[x.name for x in INVARIANT_DEFINITIONS] if invariants else sorted(values)
    return {name:describe(values[name]) for name in names}

def _stability(a,b):
    rows={}; changes=[]
    for name in sorted(set(a)|set(b)):
        acv=(a.get(name) or {}).get("cv"); bcv=(b.get(name) or {}).get("cv")
        change=None if acv in (None,0) or bcv is None else (acv-bcv)/acv
        rows[name]={"a_cv":acv,"b_cv":bcv,"relative_cv_improvement":change,
                    "a_mad":(a.get(name) or {}).get("mad"),"b_mad":(b.get(name) or {}).get("mad")}
        if change is not None: changes.append(change)
    return {"per_metric":rows,"comparable_count":len(changes),"improved_count":sum(x>0 for x in changes),
            "degraded_count":sum(x<0 for x in changes),"median_relative_cv_improvement":describe(changes)["median"]}

def _resolution(records):
    vals={k:[] for k in ("face_width_px","face_height_px","face_area_ratio")}
    for r in records:
        metrics=r["pfr"].get("quality",{}).get("metrics",{})
        for key in vals:
            v=_num(r["item"].get(key)); v=v if v is not None else _num(metrics.get(key))
            if v is not None: vals[key].append(v)
    return {k:describe(v) for k,v in vals.items()}

def _pose(records):
    vals={k:[] for k in ("roll_degrees","yaw_proxy","pitch_proxy","blur_score")}
    for r in records:
        p=r["pfr"]; pose=p.get("canonical_mesh",{}).get("pose",{}); q=p.get("quality",{}).get("metrics",{})
        choices={"roll_degrees":(pose.get("roll_degrees"),q.get("roll_degrees")),"yaw_proxy":(pose.get("yaw_proxy"),q.get("yaw_offset_ratio")),
                 "pitch_proxy":(pose.get("pitch_proxy"),q.get("pitch_offset_ratio")),"blur_score":(q.get("blur_score"),)}
        for k,xs in choices.items():
            v=next((_num(x) for x in xs if _num(x) is not None),None)
            if v is not None: vals[k].append(v)
    out={k:describe(v) for k,v in vals.items()}
    for k in ("roll_degrees","yaw_proxy","pitch_proxy"): out[k]["max_abs"]=max(map(abs,vals[k]),default=None)
    out["note"]="yaw_proxy and pitch_proxy are dimensionless proxies, not degrees"; return out

def _temporal(d):
    fps=_num(d["manifest"].get("video_source",{}).get("fps")); points=[]
    for r in d["records"]:
        i=r["item"]; frame=_num(i.get("frame_index")); sec=_num(i.get("timestamp_seconds"))
        if sec is None and frame is not None and fps: sec=frame/fps
        if sec is not None or frame is not None: points.append((sec,frame))
    points.sort(key=lambda x:(x[0] if x[0] is not None else math.inf,x[1] if x[1] is not None else math.inf))
    secs=[b[0]-a[0] for a,b in zip(points,points[1:]) if a[0] is not None and b[0] is not None]
    frames=[b[1]-a[1] for a,b in zip(points,points[1:]) if a[1] is not None and b[1] is not None]; dup=sum(x<.5 for x in secs)
    return {"time_distance_seconds":describe(secs),"frame_distance":describe(frames),"duplicate_threshold_seconds":.5,
            "duplicate_candidate_count":dup,"duplicate_candidate_rate":dup/len(secs) if secs else None,
            "timestamp_source":"manifest timestamp or frame_index/video_fps fallback"}

def _lic(d):
    try:
        x=build_lic_stability_report(str(d["root"] / "pfr"),top=10000); rank=x.get("ranking",[])
        return {"status":"ok","preferred_base":x.get("preferred_base"),"base_stability":x.get("base_stability"),"ranking":rank,"top_5":rank[:5]}
    except Exception as e: return {"status":"unavailable","reason":str(e),"ranking":[],"top_5":[]}

def compare_datasets(dataset_a,dataset_b,*,label_a="fixed_step",label_b="quality_profile",force=False):
    a,b=_archive(dataset_a),_archive(dataset_b); validation=_validate(a,b)
    if validation["status"]=="invalid_comparison" and not force: raise ComparisonValidationError("Critical controls differ: "+", ".join(validation["critical_mismatches"]))
    ar,br=_records(a),_records(b); qa,qb=_quality(a),_quality(b); ma,mb=_stats(ar),_stats(br); ia,ib=_stats(ar,True),_stats(br,True)
    ta,tb=_temporal(a),_temporal(b); la,lb=_lic(a),_lic(b); ist,mst=_stability(ia,ib),_stability(ma,mb)
    sensitivity={"A_all":{"count":len(ar),"measurements":ma,"invariants":ia},"B_all":{"count":len(br),"measurements":mb,"invariants":ib}}
    for key,status in (("B_passed_only","passed"),("B_warning_only","warning")):
        rows=_records(b,status); sensitivity[key]={"count":len(rows),"measurements":_stats(rows),"invariants":_stats(rows,True)}
    qcomp={c:{"a_rate":qa["issue_codes"][c]["rate"],"b_rate":qb["issue_codes"][c]["rate"],
       "difference_b_minus_a":qb["issue_codes"][c]["rate"]-qa["issue_codes"][c]["rate"],
       "relative_change":relative_change(qa["issue_codes"][c]["rate"],qb["issue_codes"][c]["rate"])} for c in qa["issue_codes"]}
    valid=validation["status"]!="invalid_comparison"; da,db=ta["duplicate_candidate_rate"],tb["duplicate_candidate_rate"]
    conclusion={"scientific_status":"exploratory; association is not proof of causality",
      "quality_cleaner":valid and qb["statuses"].get("passed",0)/max(1,qb["total"])>qa["statuses"].get("passed",0)/max(1,qa["total"]),
      "invariants_more_stable":valid and ist["improved_count"]>ist["degraded_count"],
      "measurements_more_stable":valid and mst["improved_count"]>mst["degraded_count"],
      "lic_ranking_changed":la["top_5"]!=lb["top_5"],"fewer_near_duplicates":valid and da is not None and db is not None and db<da,
      "recommended_for_frontal_neutral":valid and ist["improved_count"]>ist["degraded_count"] and (db is None or db<=.25),
      "strongest_differences":sorted(ist["per_metric"],key=lambda k:abs(ist["per_metric"][k]["relative_cv_improvement"] or 0),reverse=True)[:5],
      "risks":["single source video","non-random selection","proxy pose metrics","small subsets may be unstable"]+(["comparison controls failed"] if not valid else []),
      "sensitivity_note":"passed-only subset is empty; result unavailable" if sensitivity["B_passed_only"]["count"]==0 else "passed-only and warning-only subsets reported separately"}
    return {"schema":{"name":"orion-ab-comparison","version":"1.0"},"created_at":datetime.now(timezone.utc).isoformat(),
      "provenance":{"python":sys.version.split()[0],"platform":platform.platform(),"dataset_a":str(a["root"]),"dataset_b":str(b["root"])},
      "labels":{"a":label_a,"b":label_b},"control_validation":validation,"selection_parameters":{"a":_selection(a,"A"),"b":_selection(b,"B")},
      "quality":{"a":qa,"b":qb,"comparison":qcomp},"effective_resolution":{"a":_resolution(ar),"b":_resolution(br)},
      "measurements":{"a":ma,"b":mb,"stability":mst},"invariants":{"registry":[x.name for x in INVARIANT_DEFINITIONS],"a":ia,"b":ib,"stability":ist,
      "note":"computed through ratio_engine; candidates are not universal validated invariants"},"pose_and_blur":{"a":_pose(ar),"b":_pose(br)},
      "temporal_deduplication":{"a":ta,"b":tb},"lic":{"a":la,"b":lb,"top5_changed":la["top_5"]!=lb["top_5"]},
      "sensitivity_analysis":sensitivity,"conclusion":conclusion}
