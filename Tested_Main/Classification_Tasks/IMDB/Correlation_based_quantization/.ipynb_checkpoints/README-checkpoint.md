# BERT-IMDB Mixed-Precision Post-Training Quantization

This directory contains scripts for applying sensitivity-based mixed-precision quantization to BERT models fine-tuned on IMDB sentiment classification.

## Dataset Split Strategy (Avoiding Data Leakage)

**CRITICAL:** To ensure valid PTQ results, we use **different data splits** for calibration and evaluation:

| Script | Dataset Split | Purpose | Size |
|--------|--------------|---------|------|
| `sensitivity_BERT_IMDB.py` | **TRAIN[:5000]** | Calibration (compute sensitivities) | 5,000 examples |
| `quantize_eval_BERT_IMDB.py` | **TEST** | Evaluation (report metrics) | 25,000 examples |

### Why This Matters:
- ✅ **No data leakage**: Sensitivities computed on train, evaluation on test
- ✅ **Standard PTQ practice**: Matches literature (GPTQ, SmoothQuant, AWQ)
- ✅ **Valid results**: Test accuracy represents true generalization

---

## Model Information

**Pre-trained Model:** `textattack/bert-base-uncased-imdb`
- BERT-base fine-tuned on IMDB dataset
- Expected accuracy: ~93% on test set (before quantization)
- 12 layers, 768 hidden dim, 110M parameters

---

## Step 1: Compute Layer Sensitivities

Computes PWCCA-based sensitivities on **TRAIN set (first 5000 examples)**:

```bash
python sensitivity_BERT_IMDB.py
```

**Output:**
- `Sensitivities/layer_sensitivity_BERT_IMDB.json` - Sensitivity values per layer
- `Sensitivities/layer_sensitivity_BERT_IMDB_runtime_*.txt` - Human-readable log

**Expected Runtime:** ~10-15 minutes on 4 GPUs

---

## Step 2: Quantize and Evaluate

Applies mixed-precision quantization and evaluates on **TEST set**:

```bash
python quantize_eval_BERT_IMDB.py
```

**What it does:**
1. ✅ Loads fine-tuned BERT model
2. ✅ Evaluates FP32 model on test set (baseline metrics)
3. ✅ Loads sensitivities from Step 1
4. ✅ Applies K-means clustering to assign bit-widths (2, 4, 8, 16)
5. ✅ Quantizes model layers
6. ✅ Evaluates quantized model on test set
7. ✅ Reports compression ratio, accuracy drop, performance metrics

**Output:**
- `Evaluation/quantize_eval_BERT_IMDB.txt` - Complete results

**Expected Runtime:** ~20-30 minutes on 4 GPUs

---

## Expected Results

### Baseline (FP32):
- **Accuracy:** ~0.93
- **Precision:** ~0.93
- **Recall:** ~0.93
- **F1-Score:** ~0.93
- **Model Size:** ~420 MB

### After Mixed-Precision PTQ:
- **Accuracy:** ~0.90-0.91 (2-3% drop)
- **Compression Ratio:** ~5-6x
- **Model Size:** ~70-80 MB
- **Bit Allocation:** Sensitive layers → 16-bit, Less sensitive → 2-bit

---

## Key Differences from OPT-HellaSwag

| Aspect | OPT-HellaSwag | BERT-IMDB |
|--------|---------------|-----------|
| **Model Type** | Causal LM | Sequence Classification |
| **Task** | Multiple Choice | Binary Sentiment |
| **Calibration** | HellaSwag val ❌ (leakage) | IMDB train[:5000] ✅ |
| **Evaluation** | HellaSwag val ❌ (same as calib) | IMDB test ✅ |
| **Metrics** | Accuracy only | Accuracy, Precision, Recall, F1 |
| **Layer Pattern** | `.layers.X.` (OPT) | `.layer.X.` (BERT) |
| **Max Length** | 128 tokens | 512 tokens |

---

## Directory Structure

```
IMDB/Correlation_based_quantization/
├── sensitivity_BERT_IMDB.py          # Step 1: Compute sensitivities
├── quantize_eval_BERT_IMDB.py        # Step 2: Quantize & evaluate
├── README.md                          # This file
├── Sensitivities/
│   ├── layer_sensitivity_BERT_IMDB.json
│   └── layer_sensitivity_BERT_IMDB_runtime_*.txt
└── Evaluation/
    └── quantize_eval_BERT_IMDB.txt
```

---

## Literature Reference

This implementation follows standard PTQ practices from:
- **GPTQ** (Frantar et al., 2022): Uses separate calibration data
- **SmoothQuant** (Xiao et al., 2023): 512-1000 calibration samples
- **AWQ** (Lin et al., 2023): Small calibration set from training data

---

## Notes

1. **HuggingFace Cache:** All models/datasets cached to `/pscratch/sd/a/ananda/.cache/huggingface/`
2. **GPU Usage:** Scripts automatically use all available GPUs with DataParallel
3. **Reproducibility:** Seed fixed to 42 for deterministic results
4. **Memory:** BERT-base fits easily on 1 GPU, but 4 GPUs speed up processing




python sensitivity_BERT_IMDB.py
# or explicitly:
python sensitivity_BERT_IMDB.py --method pwcca

python sensitivity_BERT_IMDB.py --method svcca --topk 20


python quantize_eval_BERT_IMDB.py
# or explicitly:
python quantize_eval_BERT_IMDB.py --allocation kmeans --sensitivity_method pwcca


python quantize_eval_BERT_IMDB.py --allocation percentile --sensitivity_method pwcca


# First compute SVCCA sensitivities:
python sensitivity_BERT_IMDB.py --method svcca --topk 20

# Then run quantization with percentile allocation:
python quantize_eval_BERT_IMDB.py --allocation percentile --sensitivity_method svcca