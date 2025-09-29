#!/usr/bin/env python3
import os, json, argparse, sys
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

def safe_get(metric_block, keys):
    for k in keys:
        if k in metric_block and metric_block[k] is not None:
            return metric_block[k]
    return None

def load_metrics(metrics_path, report_path):
    METRICS = json.load(open(metrics_path))
    REPORT  = json.load(open(report_path))
    debug = {"val_keys": [], "test_keys": []}

    val = METRICS.get("val", {}) or {}
    test = METRICS.get("test", {}) or {}

    # First try to read from metrics.json
    val_micro = safe_get(val, ["f1_micro", "eval_f1_micro"])
    val_macro = safe_get(val, ["f1_macro", "eval_f1_macro"])
    val_weighted = safe_get(val, ["f1_weighted", "eval_f1_weighted"])
    val_loss = safe_get(val, ["loss", "eval_loss"])

    test_micro = safe_get(test, ["f1_micro", "eval_f1_micro"])
    test_macro = safe_get(test, ["f1_macro", "eval_f1_macro"])
    test_weighted = safe_get(test, ["f1_weighted", "eval_f1_weighted"])
    test_loss = safe_get(test, ["loss", "eval_loss"])

    # If still missing, backfill from classification_report.json
    # sklearn's output_dict includes 'micro avg', 'macro avg', 'weighted avg'
    def backfill_from_report(name, current_micro, current_macro, current_weighted):
        if current_micro is None or current_macro is None or current_weighted is None:
            micro = REPORT.get("micro avg", {}).get("f1-score")
            macro = REPORT.get("macro avg", {}).get("f1-score")
            weighted = REPORT.get("weighted avg", {}).get("f1-score")
            return (
                current_micro if current_micro is not None else micro,
                current_macro if current_macro is not None else macro,
                current_weighted if current_weighted is not None else weighted,
            )
        return current_micro, current_macro, current_weighted

    val_micro, val_macro, val_weighted = backfill_from_report("val", val_micro, val_macro, val_weighted)
    test_micro, test_macro, test_weighted = backfill_from_report("test", test_micro, test_macro, test_weighted)

    debug["val_keys"] = list(val.keys())
    debug["test_keys"] = list(test.keys())

    return METRICS, (val_micro, val_macro, val_weighted, val_loss), (test_micro, test_macro, test_weighted, test_loss), debug

def build_chart(report_path, chart_path, label_names):
    REPORT = json.load(open(report_path))
    rows = []
    for lbl in label_names:
        stats = REPORT.get(lbl, {})
        if isinstance(stats, dict) and "f1-score" in stats:
            rows.append({"label": lbl,
                         "precision": stats.get("precision", 0.0),
                         "recall": stats.get("recall", 0.0),
                         "f1": stats.get("f1-score", 0.0),
                         "support": stats.get("support", 0)})
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    top = df.head(10); bottom = df.tail(10)

    plt.figure(figsize=(7,5))
    labels = list(top["label"]) + ["..."] + list(bottom["label"])
    values = list(top["f1"]) + [None] + list(bottom["f1"])
    x = list(range(len(values)))
    sep_idx = labels.index("...")
    plt.bar(x[:sep_idx], values[:sep_idx])
    plt.bar(x[sep_idx+1:], values[sep_idx+1:])
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.title("Per-label F1 (Top-10 & Bottom-10)")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=200)
    plt.close()
    return chart_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_dir", required=True)
    ap.add_argument("--pdf_path", default="./goemotions_baseline_report.pdf")
    args = ap.parse_args()

    metrics_path = os.path.join(args.outputs_dir, "metrics.json")
    report_path  = os.path.join(args.outputs_dir, "classification_report.json")
    eff_path     = os.path.join(args.outputs_dir, "efficiency_snapshot.json")
    for p in [metrics_path, report_path, eff_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    METRICS, VALS, TESTS, debug = load_metrics(metrics_path, report_path)
    EFF = json.load(open(eff_path))

    label_names = METRICS.get("label_names", [])
    cfg = METRICS.get("args", {})

    chart_path = os.path.join(args.outputs_dir, "label_f1_chart.png")
    build_chart(report_path, chart_path, label_names)

    styles = getSampleStyleSheet()
    title, h2, body = styles["Title"], styles["Heading2"], styles["BodyText"]
    doc = SimpleDocTemplate(args.pdf_path, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=1.5*cm, bottomMargin=1.5*cm)
    story = []
    story.append(Paragraph("GoEmotions Baseline Report", title))
    story.append(Paragraph(datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), body))
    story.append(Spacer(1,12))

    summary = f"""
    <b>Model:</b> {cfg.get('model_name')}<br/>
    <b>Dataset:</b> {cfg.get('dataset_name')} ({cfg.get('dataset_config')})<br/>
    <b>Training:</b> epochs={cfg.get('epochs', cfg.get('num_train_epochs', 3))},
    batch_size={cfg.get('batch_size', cfg.get('per_device_train_batch_size', 32))},
    max_length={cfg.get('max_length', 128)},
    threshold={cfg.get('eval_threshold', 0.5)}<br/>
    """
    story.append(Paragraph("Run Summary", h2))
    story.append(Paragraph(summary, body)); story.append(Spacer(1,8))

    (val_micro, val_macro, val_weighted, val_loss) = VALS
    (t_micro, t_macro, t_weighted, t_loss) = TESTS

    def fmt(x):
        try:
            return f"{float(x):.4f}"
        except Exception:
            return "nan"

    tbl = [
        ["", "F1 (micro)", "F1 (macro)", "F1 (weighted)", "Loss"],
        ["Validation", fmt(val_micro), fmt(val_macro), fmt(val_weighted), fmt(val_loss)],
        ["Test", fmt(t_micro), fmt(t_macro), fmt(t_weighted), fmt(t_loss)],
    ]
    table = Table(tbl, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (1,1), (-1,-1), "CENTER"),
    ]))
    story.append(Paragraph("Evaluation Metrics", h2)); story.append(table); story.append(Spacer(1,8))

    eff_txt = f"""
    <b>Trainable parameters:</b> {EFF.get('trainable_params','?'):,}<br/>
    <b>Avg latency (ms) per batch of 32:</b> {EFF.get('avg_latency_ms_per_batch32','?'):.2f}
    """
    story.append(Paragraph("Efficiency Snapshot", h2))
    story.append(Paragraph(eff_txt, body)); story.append(Spacer(1,8))

    if os.path.exists(chart_path):
        story.append(Paragraph("Per-label F1 — Top & Bottom 10", h2))
        story.append(Image(chart_path, width=16*cm, height=10*cm)); story.append(Spacer(1,8))

    # Minimal debug footer so you know what keys were seen
    dbg = f"<b>Debug:</b> val keys={debug['val_keys']}, test keys={debug['test_keys']}"
    story.append(Paragraph("Debug", h2)); story.append(Paragraph(dbg, body))

    notes = """
    <b>Notes:</b><br/>
    • Accepts either eval_* or plain metric keys; if missing, backfills from classification_report.json.<br/>
    • Scores use a fixed threshold (default 0.5). Threshold sweeps can improve macro-F1.<br/>
    • Latency is a simple forward pass; serving latency varies by hardware and batch size.
    """
    story.append(Paragraph("Notes", h2))
    story.append(Paragraph(notes, body))
    doc.build(story)
    print("Wrote PDF to:", args.pdf_path)

if __name__ == "__main__":
    main()
