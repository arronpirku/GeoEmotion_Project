"""
GoEmotions: Raw vs Simplified — Comparison Report (PDF)

Usage (PowerShell on Windows):
  python make_report_v2.py --roots .\simplified_models .\raw_models --pdf_path .\goemotions_raw_vs_simplified_report.pdf

Usage (bash/macOS/Linux):
  python make_report_v2.py --roots ./simplified_models ./raw_models --pdf_path ./goemotions_raw_vs_simplified_report.pdf
"""

import os, json, glob, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table as RLTable, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

SCHEMES = ("Raw", "Simplified")

SIMPLE_MODEL_NAMES = {
    "ALBERT-base-v2": "ALBERT",
    "DistilBERT": "DistilBERT",
    "MobileBERT": "MobileBERT",
    "MiniLM-L6-H384": "MiniLM",
}
def _simple(name: str) -> str:
    return SIMPLE_MODEL_NAMES.get(name, name)

def read_json_safe(path):
    try:
        return json.load(open(path, "r"))
    except Exception:
        return None

def try_read_metrics(run_dir: Path):
    m = read_json_safe(run_dir / "metrics.json")
    if m is not None:
        val_macro = m.get("val", {}).get("eval_f1_macro")
        test_macro = m.get("test", {}).get("eval_f1_macro")
        return val_macro, test_macro
    
    rep = read_json_safe(run_dir / "classification_report.json")
    if isinstance(rep, dict) and "macro avg" in rep and "f1-score" in rep["macro avg"]:
    
        return None, rep["macro avg"]["f1-score"]
    return None, None

def friendly_model_name(raw: str):
    mapping = {
        "distilbert-base-uncased": "DistilBERT",
        "albert-base-v2": "ALBERT-base-v2",
        "google/mobilebert-uncased": "MobileBERT",
        "nreimers/MiniLM-L6-H384-uncased": "MiniLM-L6-H384",
        "mobilebert-uncased": "MobileBERT",
        "MiniLM-L6-H384-uncased": "MiniLM-L6-H384",
        "minilm": "MiniLM-L6-H384",
        "mobilebert": "MobileBERT",
        "distilbert": "DistilBERT",
        "albert": "ALBERT-base-v2",
    }
    return mapping.get(raw, raw)

def infer_scheme_from_path(p: Path):
    s = p.as_posix().lower()
    if "raw" in s:
        return "Raw"
    if "simplified" in s or "simp" in s:
        return "Simplified"
    return None

def discover_run_dirs(roots):
    """Find directories that contain a run (look for run_manifest.json OR metrics.json)."""
    run_dirs = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        
        cands = list(root.glob("*")) + list(root.glob("*/*"))
        for d in cands:
            if not d.is_dir():
                continue
            if (d / "run_manifest.json").exists() or (d / "metrics.json").exists():
                run_dirs.append(d)
    return sorted(set(run_dirs))

def load_run_dir(run_dir: Path):
    out = {"run_dir": str(run_dir)}
    man = read_json_safe(run_dir / "run_manifest.json")

    if man:
        out["model"] = Path(man.get("model_id", "")).name or man.get("model_id", "")
        out["dataset_config"] = man.get("dataset_config")
        out["label_type"] = man.get("label_type")
        out["num_labels"] = man.get("num_labels")
        out["label_names"] = man.get("label_names")
        out["train_args"] = man.get("train_args", {})
    else:

        out["model"] = run_dir.name
        out["dataset_config"] = None
        out["label_type"] = None
        out["num_labels"] = None
        out["label_names"] = None
        out["train_args"] = {}


    scheme_from_name = infer_scheme_from_path(run_dir)
    if out.get("dataset_config") in ("raw", "simplified"):
        out["scheme"] = "Raw" if out["dataset_config"] == "raw" else "Simplified"
    elif scheme_from_name:
        out["scheme"] = scheme_from_name
    else:
        out["scheme"] = None


    val_macro, test_macro = try_read_metrics(run_dir)
    out["val_macro"] = val_macro
    out["test_macro"] = test_macro


    eff = read_json_safe(run_dir / "efficiency_snapshot.json")
    if eff:
       
        tp = eff.get("trainable_params")
        out["params_M"] = (float(tp) / 1e6) if tp is not None else None

       
        cpu_lat = eff.get("avg_latency_ms_per_batch32_cpu")
        legacy  = eff.get("avg_latency_ms_per_batch32") 
        gpu_lat = eff.get("avg_latency_ms_per_batch32_gpu")

        if cpu_lat is None:
            cpu_lat = legacy

        out["latency_ms_b32"]      = cpu_lat
        out["latency_ms_b32_gpu"]  = gpu_lat
    else:
        out["params_M"] = None
        out["latency_ms_b32"] = None
        out["latency_ms_b32_gpu"] = None

    ptq_t = read_json_safe(run_dir / "quant_results" / "ptq_eval.json")
    if ptq_t:
        out["ptq_split"]      = ptq_t.get("split")
        out["ptq_thr"]        = ptq_t.get("threshold")
        out["ptq_fp32_macro"] = (ptq_t.get("fp32") or {}).get("f1_macro")
        out["ptq_int8_macro"] = (ptq_t.get("int8_dynamic") or {}).get("f1_macro")
        out["ptq_fp32_lat_s"] = (ptq_t.get("fp32") or {}).get("latency_sec")
        out["ptq_int8_lat_s"] = (ptq_t.get("int8_dynamic") or {}).get("latency_sec")

    ptq_b = read_json_safe(run_dir / "quant_baseline" / "ptq_eval.json")
    if ptq_b:
        out["ptq_base_split"]      = ptq_b.get("split")
        out["ptq_base_thr"]        = ptq_b.get("threshold")
        out["ptq_base_fp32_macro"] = (ptq_b.get("fp32") or {}).get("f1_macro")
        out["ptq_base_int8_macro"] = (ptq_b.get("int8_dynamic") or {}).get("f1_macro")
        out["ptq_base_fp32_lat_s"] = (ptq_b.get("fp32") or {}).get("latency_sec")
        out["ptq_base_int8_lat_s"] = (ptq_b.get("int8_dynamic") or {}).get("latency_sec")

    return out

def build_dataframe(roots):
    rows = [load_run_dir(d) for d in discover_run_dirs(roots)]
    df = pd.DataFrame(rows)

    df["model"] = df["model"].map(lambda x: friendly_model_name(str(x)))

    df["scheme"] = df["scheme"].map(lambda s: s if s in SCHEMES else None)
    return df


def style_best_second(df, col_modes):
    """
    Build a style matrix for ReportLab tables.
    col_modes: dict column_name -> "max" (higher is better) or "min" (lower is better)
    """
    styles = [["" for _ in df.columns] for _ in range(len(df))]
    if not col_modes:
        return styles
    for col, mode in col_modes.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() == 0:
            continue
        ascending = (mode == "min")
        ranks = series.rank(method="min", ascending=ascending)
        
        for i in range(len(df)):
            if pd.isna(series.iloc[i]):
                continue
            if ranks.iloc[i] == 1:
                styles[i][df.columns.get_loc(col)] = "bold"
            elif ranks.iloc[i] == 2:
                styles[i][df.columns.get_loc(col)] = "italic"
    return styles

def render_table_rl(df, title, number=None, col_modes=None, table_width=500, col_widths=None):
    """
    Render a pandas df as a ReportLab table with bold/italic best/second styling.
    table_width: total width in points; col_widths: optional {column_name: width_points}
    """
    headers = list(df.columns)
    ncols = len(headers)

    default_w = table_width / max(ncols, 1)
    widths = [default_w] * ncols
    if isinstance(col_widths, dict):
        for i, col in enumerate(headers):
            if col in col_widths:
                widths[i] = col_widths[col]
        total = sum(widths)
        if total > table_width:
            scale = table_width / total
            widths = [w * scale for w in widths]

    data = [headers] + df.values.tolist()
    t = RLTable(data, hAlign="LEFT", colWidths=widths)
    ts = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('LINEBELOW', (0,0), (-1,0), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('BOX', (0,0), (-1,-1), 0.5, colors.black),
        ('LEFTPADDING',(0,0),(-1,-1),4),
        ('RIGHTPADDING',(0,0),(-1,-1),4),
        ('TOPPADDING',(0,0),(-1,-1),2),
        ('BOTTOMPADDING',(0,0),(-1,-1),2),
    ])
    styles = style_best_second(df, col_modes)
    for r in range(len(df)):
        for c in range(len(df.columns)):
            s = styles[r][c]
            if s == "bold":
                ts.add('FONTNAME', (c, r+1), (c, r+1), 'Helvetica-Bold')
            elif s == "italic":
                ts.add('FONTNAME', (c, r+1), (c, r+1), 'Helvetica-Oblique')
    t.setStyle(ts)
    caption = f"Table {number}. {title}" if number is not None else title
    return caption, t

# ------------------------- Plots -------------------------

def pivot_safe(df, value_col):
    p = df.pivot_table(index="model", columns="scheme", values=value_col, aggfunc="first")
    for s in SCHEMES:
        if s not in p.columns:
            p[s] = np.nan
    return p.reindex(columns=list(SCHEMES))

def fig_macro_delta(df, out_png, number):
    """
    Δ Macro-F1 (Simplified - Raw) for Val and Test.
    Uses the same pivots as Table 1 to avoid inconsistency.
    """
    pv_val  = pivot_safe(df, "val_macro")
    pv_test = pivot_safe(df, "test_macro")

    have_val  = pv_val[list(SCHEMES)].notna().any().all()
    have_test = pv_test[list(SCHEMES)].notna().any().all()

    if not (have_val or have_test):
        plt.figure(figsize=(6,3)); plt.axis("off")
        plt.text(0.5, 0.5, f"Figure {number} skipped:\nneed Raw & Simplified to plot Δ.", ha="center", va="center")
        plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()
        return

    base = pv_test if have_test else pv_val
    models = base.index
    x = np.arange(len(models))
    plt.figure(figsize=(6,3))
    plt.axhline(0, color="k", linewidth=1)

    if have_val:
        dv = (pv_val.loc[models, "Simplified"] - pv_val.loc[models, "Raw"]).values
        plt.bar(x-0.2, dv, width=0.4, label="Validation Δ")
    if have_test:
        dt = (pv_test.loc[models, "Simplified"] - pv_test.loc[models, "Raw"]).values
        plt.bar(x+0.2, dt, width=0.4, label="Test Δ")

    plt.xticks(x, models, rotation=0)
    plt.ylabel("Macro-F1 (Simplified − Raw)")
    plt.title(f"Figure {number}. Macro-F1 Δ by Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def fig_latency_vs_macro(df, out_png, number):
    from matplotlib.lines import Line2D
    colors = {"Raw":"#1f77b4","Simplified":"#ff7f0e"}
    markers= {"DistilBERT":"o","ALBERT-base-v2":"s","MobileBERT":"^","MiniLM-L6-H384":"D"}

    sub = df.dropna(subset=["test_macro","latency_ms_b32"]).copy()
    plt.figure(figsize=(6,3))
    for _, r in sub.iterrows():
        plt.scatter(r["test_macro"], r["latency_ms_b32"],
                    c=colors.get(r["scheme"], "#888"),
                    marker=markers.get(r["model"], "o"),
                    s=70)
    color_handles = [Line2D([0],[0], marker='o', color='w',
                            markerfacecolor=colors[s], markersize=8, label=s)
                     for s in ["Raw","Simplified"] if s in sub["scheme"].unique()]
    marker_handles = [Line2D([0],[0], marker=m, color='k', linestyle='None', label=mdl)
                      for mdl, m in markers.items() if mdl in sub["model"].unique()]
    plt.legend(handles=color_handles+marker_handles, loc='best', fontsize=8, title="Legend")
    plt.xlabel("Macro-F1 (Test)")
    plt.ylabel("Latency (ms / batch=32)")
    plt.title(f"Figure {number}. Latency vs Macro-F1 (color=scheme, marker=model)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def fig_test_bars(df, out_png, number):
    """Side-by-side bars: Raw vs Simplified (Test Macro-F1) per model."""
    pv = df.pivot_table(index="model", columns="scheme", values="test_macro", aggfunc="first")
    for s in SCHEMES:
        if s not in pv.columns: pv[s] = np.nan
    pv = pv.reindex(columns=list(SCHEMES))
    pv = pv.sort_index()

    x = np.arange(len(pv.index))
    width = 0.35
    plt.figure(figsize=(6,3))
    plt.bar(x - width/2, pv["Raw"].values, width=width, label="Raw")
    plt.bar(x + width/2, pv["Simplified"].values, width=width, label="Simplified")
    plt.xticks(x, pv.index, rotation=0)
    plt.ylabel("Macro-F1 (Test)")
    plt.title(f"Figure {number}. Test Macro-F1 by Model and Scheme")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

# ------------------------- Quant table -------------------------

def build_quant_table(df):
    q = df.dropna(subset=["ptq_fp32_macro","ptq_int8_macro"]).copy()
    if q.empty: return None
    q = q[["model","scheme","ptq_fp32_macro","ptq_int8_macro","ptq_fp32_lat_s","ptq_int8_lat_s"]]
    q["Δ Macro"] = q["ptq_int8_macro"] - q["ptq_fp32_macro"]   # shorter header so it fits
    q["Speedup×"] = q["ptq_fp32_lat_s"] / q["ptq_int8_lat_s"]
    q = q.rename(columns={
        "ptq_fp32_macro":"FP32 Macro",
        "ptq_int8_macro":"INT8 Macro",
        "ptq_fp32_lat_s":"FP32 Lat (s)",
        "ptq_int8_lat_s":"INT8 Lat (s)",
    }).sort_values(["scheme","model"]).reset_index(drop=True)
    return q


def make_exec_summary(df):
    args_list = df["train_args"].dropna().tolist()
    epochs = lr = wd = bs = None
    for a in args_list:
        if not isinstance(a, dict): continue
        epochs = epochs or a.get("epochs")
        lr     = lr or a.get("lr")
        wd     = wd or a.get("weight_decay")
        bs     = bs or a.get("batch_size")
    return (
        "<b>Executive Summary.</b> We trained and evaluated four encoders—DistilBERT, "
        "ALBERT-base-v2, MobileBERT, and MiniLM-L6-H384—on the GoEmotions dataset under two label "
        "schemes: <b>Raw</b> (28-class multi-label) and <b>Simplified</b> (single-label). "
        f"Unless noted otherwise, runs used {epochs or 3} epochs, batch={bs or 32}, lr={lr or '5e-5'}, "
        f"weight decay={wd or 0.01}. We report Macro-F1 on validation and test, parameter count, "
        "and CPU forward-latency (ms per batch=32). We also evaluate dynamic INT8 post-training "
        "quantization (FP32 vs INT8 accuracy and latency)."
    )

def make_defs_text():
    return (
        "<b>Definitions.</b> Macro-F1 = unweighted mean of per-class F1. Micro-F1 = F1 from global "
        "TP/FP/FN. Weighted-F1 = class-frequency-weighted mean. <i>Raw</i>: sigmoid + fixed threshold "
        "(tuned on validation). <i>Simplified</i>: softmax argmax. Latency: forward-pass time per "
        "batch=32 on CPU (tokenization not timed). <i>Params (M)</i>: trainable parameters in millions."
    )

def make_assessment(df):
    return (
        "<b>Assessment.</b> Across all four encoders, the Simplified labeling scheme is decisively "
        "better than Raw, improving test Macro-F1 by roughly +0.16–0.28 per model. On Simplified, "
        "MobileBERT delivers the highest accuracy (~0.47 Macro-F1) at moderate latency, making it "
        "the best overall choice. MiniLM is the fastest on CPU (≈80–90 ms/b32) with a predictable "
        "accuracy trade-off (~0.35 Macro-F1), while DistilBERT sits in the middle on both axes; "
        "ALBERT is accurate but comparatively slow, and INT8 causes a severe drop on Simplified. "
        "We recommend MobileBERT for the best accuracy–latency balance, and MiniLM when latency dominates."
    )


# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True,
                    help="One or more root folders containing model run folders (e.g., simplified_models raw_models)")
    ap.add_argument("--pdf_path", required=True)
    args = ap.parse_args()

    df = build_dataframe(args.roots)

    # --- Table 1: Macro-F1 (Val/Test) ---
    t1 = (df.pivot_table(index="model", columns="scheme",
                         values=["val_macro","test_macro"], aggfunc="first"))

    t1 = t1.reindex(columns=pd.MultiIndex.from_product([["val_macro","test_macro"], list(SCHEMES)]))
    t1.columns = [f"{a} {b}" for a,b in t1.columns]
    t1 = (t1.rename(columns={"val_macro Raw":"Raw Val","val_macro Simplified":"Simp Val",
                             "test_macro Raw":"Raw Test","test_macro Simplified":"Simp Test"})
              .reset_index())
    t1["model"] = t1["model"].map(_simple)
    t1_modes = {c: "max" for c in ["Raw Val","Simp Val","Raw Test","Simp Test"] if c in t1.columns}

    # --- Table 2: Efficiency (Params, Latency) ---
    t2 = (df.pivot_table(index="model", columns="scheme",
                         values=["params_M","latency_ms_b32"], aggfunc="first"))
    t2 = t2.reindex(columns=pd.MultiIndex.from_product([["params_M","latency_ms_b32"], list(SCHEMES)]))
    t2.columns = [f"{a} {b}" for a,b in t2.columns]
    t2 = (t2.rename(columns={"params_M Raw":"Params (M) Raw","params_M Simplified":"Params (M) Simplified",
                             "latency_ms_b32 Raw":"Latency Raw (ms/b32)","latency_ms_b32 Simplified":"Latency Simplified (ms/b32)"})
              .reset_index())
    t2["model"] = t2["model"].map(_simple)
    t2_modes = {}
    if "Latency Raw (ms/b32)" in t2.columns: t2_modes["Latency Raw (ms/b32)"] = "min"
    if "Latency Simplified (ms/b32)" in t2.columns: t2_modes["Latency Simplified (ms/b32)"] = "min"

    # --- Table 3: PTQ ---
    qtab = build_quant_table(df)
    if qtab is not None and not qtab.empty:
        t3 = qtab.round(3)
        t3["model"] = t3["model"].map(_simple)
        t3_modes = {"FP32 Macro":"max", "INT8 Macro":"max", "Speedup×":"max"}
    else:
        t3 = None

    # --- Figures ---
    os.makedirs("_report_figs", exist_ok=True)
    fig1 = "_report_figs/macro_delta.png"
    fig2 = "_report_figs/latency_vs_macro.png"
    fig3 = "_report_figs/test_bars.png"
    fig_macro_delta(df, fig1, number=1)
    fig_latency_vs_macro(df, fig2, number=2)
    fig_test_bars(df, fig3, number=3)

    # --- Build PDF ---
    doc = SimpleDocTemplate(args.pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    flow = []

    flow += [Paragraph("GoEmotions: Raw vs Simplified + Quantization — Model Comparisons", styles["Title"]),
             Spacer(1, 6)]

    # Intro
    exec_text = make_exec_summary(df)
    defs_text = make_defs_text()
    flow += [Paragraph(exec_text, styles["BodyText"]), Spacer(1,6),
             Paragraph(defs_text, styles["BodyText"]), Spacer(1,12)]

    # Table 1
    cap1, tab1 = render_table_rl(t1.round(3), "Macro-F1 (Validation/Test) by Model and Scheme",
                                 number=1, col_modes=t1_modes, table_width=500)
    flow += [Paragraph(cap1, styles["BodyText"]), Spacer(1,6), tab1, Spacer(1,12)]

    # Figure 1
    flow += [Image(fig1, width=480, height=260), Spacer(1,12)]

    # Table 2
    cap2, tab2 = render_table_rl(t2.round(2), "Efficiency Snapshot (Params, CPU Latency)",
                                 number=2, col_modes=t2_modes, table_width=500)
    flow += [Paragraph(cap2, styles["BodyText"]), Spacer(1,6), tab2, Spacer(1,12)]

    # Figure 2
    flow += [Image(fig2, width=480, height=260), Spacer(1,12)]

    # Figure 3
    flow += [Image(fig3, width=480, height=260), Spacer(1,12)]

    # Table 3 (PTQ, trained)
    if t3 is not None:
        narrow = {"Δ Macro": 55, "Speedup×": 55}
        cap3, tab3 = render_table_rl(
            t3, "PTQ (trained): FP32 vs INT8 Macro-F1 and Latency (per scheme)",
            number=3, col_modes=t3_modes, table_width=500, col_widths=narrow
        )
        flow += [Paragraph(cap3, styles["BodyText"]), Spacer(1,6), tab3, Spacer(1,12)]

    # Assessment
    flow += [Paragraph(make_assessment(df), styles["BodyText"])]

    doc.build(flow)
    print("Wrote PDF:", args.pdf_path)

if __name__ == "__main__":
    main()
