"""Compact human-readable PDF for an A/B comparison."""
from __future__ import annotations
from pathlib import Path
from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QColor,QFont,QGuiApplication,QPageLayout,QPageSize,QPainter,QPdfWriter

def _fmt(v):
    if v is None:return "n/a"
    if isinstance(v,bool):return "yes" if v else "no"
    if isinstance(v,float):return f"{v:.4g}"
    return str(v)

def generate_pdf(result:dict, output:str|Path)->Path:
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    app=QGuiApplication.instance() or QGuiApplication([]); writer=QPdfWriter(str(output)); writer.setResolution(120)
    writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4)); writer.setPageOrientation(QPageLayout.Orientation.Landscape)
    p=QPainter(writer); w,h=writer.width(),writer.height(); first=True
    def page(title,lines,bars=None):
        nonlocal first
        if not first: writer.newPage()
        first=False;p.fillRect(QRectF(0,0,w,h),QColor("white"));p.setPen(QColor("#17324d"));p.setFont(QFont("Arial",18,QFont.Weight.Bold));p.drawText(70,80,title)
        p.setFont(QFont("Arial",9)); y=125
        for line in lines:
            p.drawText(QRectF(70,y,w-140,32),str(line));y+=28
        if bars:
            y+=15
            for name,a,b in bars[:12]:
                p.drawText(70,y+15,name[:28]); scale=max(abs(a),abs(b),.001); x=300
                p.fillRect(QRectF(x,y,450*abs(a)/scale,10),QColor("#7896d2"));p.fillRect(QRectF(x,y+13,450*abs(b)/scale,10),QColor("#42a887"));
                p.drawText(770,y+18,f"A {_fmt(a)}   B {_fmt(b)}");y+=40
    cv=result["invariants"]["stability"]; c=result["conclusion"]; v=result["control_validation"]
    page("ORION A/B Comparison",[f"A: {result['labels']['a']}  |  B: {result['labels']['b']}",f"Control status: {v['status']}",f"Created: {result['created_at']}","Exploratory comparison; it does not establish causality."])
    q=result["quality"]; page("Archive quality",[f"A frames/PFR: {q['a']['total']}/{q['a']['pfr_count']}   statuses: {q['a']['statuses']}",f"B frames/PFR: {q['b']['total']}/{q['b']['pfr_count']}   statuses: {q['b']['statuses']}"],[(k,x['a_rate'],x['b_rate']) for k,x in q['comparison'].items()])
    page("Effective face resolution",[f"{k}: A median {_fmt(x['median'])}, B median {_fmt(result['effective_resolution']['b'][k]['median'])}" for k,x in result['effective_resolution']['a'].items()])
    page("Invariant stability",[f"Comparable: {cv['comparable_count']}; improved: {cv['improved_count']}; degraded: {cv['degraded_count']}","Lower CV indicates less relative dispersion."],[(k,x['a_cv'] or 0,x['b_cv'] or 0) for k,x in cv['per_metric'].items()])
    pb=result["pose_and_blur"];page("Pose and blur",[f"{k}: A median {_fmt(x.get('median'))}; B median {_fmt(pb['b'][k].get('median'))}" for k,x in pb['a'].items() if isinstance(x,dict)])
    t=result["temporal_deduplication"];page("Temporal diversity",[f"A median distance: {_fmt(t['a']['time_distance_seconds']['median'])} s",f"B median distance: {_fmt(t['b']['time_distance_seconds']['median'])} s",f"A/B near-duplicate rates: {_fmt(t['a']['duplicate_candidate_rate'])} / {_fmt(t['b']['duplicate_candidate_rate'])}"])
    page("LIC stability",[f"A status/base: {result['lic']['a']['status']} / {result['lic']['a'].get('preferred_base')}",f"B status/base: {result['lic']['b']['status']} / {result['lic']['b'].get('preferred_base')}",f"TOP-5 changed: {_fmt(result['lic']['top5_changed'])}"])
    page("Sensitivity analysis",[f"{k}: n={x['count']}" for k,x in result['sensitivity_analysis'].items()]+[c['sensitivity_note']])
    page("Conclusions",[f"Quality cleaner: {_fmt(c['quality_cleaner'])}",f"Invariants more stable: {_fmt(c['invariants_more_stable'])}",f"Absolute measurements more stable: {_fmt(c['measurements_more_stable'])}",f"Fewer temporal duplicate candidates: {_fmt(c['fewer_near_duplicates'])}",f"Recommend quality_profile/frontal_neutral: {_fmt(c['recommended_for_frontal_neutral'])}","Strongest differences: "+", ".join(c['strongest_differences']),"Risks: "+"; ".join(c['risks'])])
    p.end();return output
