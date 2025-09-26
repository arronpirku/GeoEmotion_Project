# Lightweight Emotion Detection with DistilBERT, ALBERT, and ModernBERT

This project explores **lightweight emotion detection** on the [GoEmotions dataset](https://github.com/google-research/google-research/tree/master/goemotions).  
The focus is on efficient transformer models (DistilBERT, ALBERT, ModernBERT) and preparing the ground for compression methods like **knowledge distillation** and **quantization**.

---

##  Features

- **Dataset handling**
  - Supports both `raw` (27 emotions + neutral) and `simplified` (6 emotions + neutral) configs.
  - Auto-creates a validation split if one is missing.
  - Works with both schema variants:
    - `labels` column (list of label IDs per sample).
    - Wide schema (binary column for each emotion).

- **Baseline models**
  - [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased)  
  - [`albert-base-v2`](https://huggingface.co/albert-base-v2)  
  - [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base)

- **Training pipeline**
  - Hugging Face `Trainer` for multi-label classification (`problem_type="multi_label_classification"`).
  - Configurable hyperparameters: epochs, batch size, learning rate, max length.
  - Evaluates micro, macro, and weighted F1.

- **Evaluation & Reporting**
  - `metrics.json` → validation/test F1 and losses.
  - `classification_report.json` → per-emotion precision/recall/F1.
  - `efficiency_snapshot.json` → trainable params + latency probe.
  - `make_report.py` → generates a polished PDF report with tables, efficiency snapshot, and per-label F1 charts.




