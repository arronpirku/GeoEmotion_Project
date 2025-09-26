Lightweight Emotion Detection with DistilBERT, ALBERT, and ModernBERT

This project explores lightweight emotion detection on the GoEmotions dataset
. The goal is to establish baselines with efficient transformer models (DistilBERT, ALBERT, ModernBERT), then apply compression and lightweighting techniques like knowledge distillation and quantization.

Features Implemented

GoEmotions dataset integration

Supports both raw (27 emotions + neutral) and simplified (6 emotions + neutral) configurations.

Auto-creates a validation split if missing.

Handles both schema variants:

list of label IDs (labels)

wide one-hot columns (one binary column per emotion).

Baseline models

distilbert-base-uncased

albert-base-v2

answerdotai/ModernBERT-base

Training pipeline

Hugging Face Trainer with multi-label classification (problem_type="multi_label_classification").

Configurable hyperparameters (epochs, batch size, learning rate, max sequence length).

Computes micro/macro/weighted F1 scores.

Evaluation & Reporting

Saves metrics.json (val/test F1 + losses).

Saves classification_report.json (per-emotion precision/recall/F1).

Logs efficiency_snapshot.json (trainable params + inference latency).

Includes a PDF report generator (make_report.py) that summarizes results, metrics tables, efficiency snapshot, and per-label F1 charts.
