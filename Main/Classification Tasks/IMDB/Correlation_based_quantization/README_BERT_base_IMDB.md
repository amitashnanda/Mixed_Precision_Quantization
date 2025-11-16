# BERT Mixed-Precision Quantization Framework for IMDB

## Overview

`BERT_base_IMDB.py` is a comprehensive, **interactive** end-to-end framework for mixed-precision quantization of BERT models on IMDB sentiment classification task. It combines **sensitivity analysis**, **layer clustering**, and **quantization evaluation** with user control at each stage.

## Features

✅ **Interactive User Prompts** - Choose options at every critical decision point
✅ **Multiple Similarity Metrics** - PWCCA, SVCCA, CKA (selectable)
✅ **Flexible Calibration** - 5k, 10k, or 25k examples (stratified or random sampling)
✅ **Multiple Clustering Strategies** - K-means, Percentile bucketing, Hierarchical
✅ **Customizable Bit Allocation** - Presets or manual specification
✅ **Comprehensive Evaluation** - Accuracy, Precision, Recall, F1, compression metrics
✅ **Detailed Logging** - All results saved with full configuration

## Quick Start

### Minimal Example (5 minutes)

```bash
python BERT_base_IMDB.py
```

Follow the interactive prompts:
- **Phase 1 (Sensitivity)**: Select "5k (fast, noisy)" → "Stratified sampling" → "PWCCA + CKA (recommended)"
- **Phase 2 (Quantization)**: Select "5k (fast)" → K-means 3 groups → [16/8/4] bits

### Full Pipeline (30 minutes)

```bash
python BERT_base_IMDB.py
```

Follow the interactive prompts:
- **Phase 1 (Sensitivity)**: Select "25k (full IMDB train)" → "Stratified sampling" → "PWCCA + CKA (recommended)"
- **Phase 2 (Quantization)**: Select "25k (full test)" → K-means 4 groups → [32/16/8/4] bits

## Detailed Guide

### Phase 1: Sensitivity Analysis

Computes how sensitive each layer is to quantization by measuring representational similarity.

#### Step 1: Calibration Set Size

| Option | Size | Speed | Stability | Use Case |
|--------|------|-------|-----------|----------|
| 5k | 5,000 examples | Fast (2-3 min) | Noisy | Quick iteration/prototyping |
| 10k | 10,000 examples | Medium (5 min) | Good | Balance |
| 25k (recommended) | Full IMDB train | Slow (10 min) | Best | Final results |

**Recommendation**: Use 5k for quick testing, then re-run with 25k for final sensitivity estimates.

#### Step 2: Sampling Strategy

- **Random**: Simple, no special logic
- **Stratified (recommended)**: Balanced 50/50 positive/negative examples
  - Better coverage of data distribution
  - More stable sensitivity estimates

#### Step 3: Similarity Metrics

| Metric | Speed | Robustness | Use |
|--------|-------|-----------|-----|
| **PWCCA** | Fast | Good default | Layer correlation weighting |
| **SVCCA** | Medium | Good, SVD-based | Dimensionality reduction |
| **CKA** | Medium | Robust, kernel-based | Verification metric |

**Recommendation**: Use **PWCCA + CKA** together
- PWCCA is fast and standard
- CKA is robust and catches representational differences
- Disagreement between them flags layers for deeper inspection

#### Example Sensitivities Output

```
layer_0: 0.123456  (embedding, usually low sensitivity)
layer_1: 0.456789
...
layer_10: 0.823456 (higher layers, usually more sensitive)
layer_11: 0.789456
```

**Interpretation**: Higher sensitivity = more important for model performance → allocate higher bits

---

### Phase 2: Quantization & Evaluation

Clusters layers by sensitivity and applies mixed-precision quantization.

#### Step 1: Evaluation Set Size

| Option | Size | Speed | Accuracy | Use Case |
|--------|------|-------|----------|----------|
| 5k | 5,000 examples | Fast (1-2 min) | Quick estimate | Iteration/debugging |
| 10k | 10,000 examples | Medium (2-3 min) | Good | Validation |
| 25k (recommended) | Full IMDB test | Slow (5 min) | Best | Final reported numbers |

**Recommendation**: Use 5k for iteration, validate with 25k final runs.

#### Step 2: Load Sensitivity File

Select the sensitivity file computed in Phase 1. If you ran multiple metrics, choose one:
- For PWCCA-only result: `layer_sensitivity_BERT_IMDB_pwcca.json`
- For SVCCA-only result: `layer_sensitivity_BERT_IMDB_svcca.json`
- For CKA-only result: `layer_sensitivity_BERT_IMDB_cka.json`

#### Step 3: Clustering Strategy

| Strategy | Method | Stability | Use |
|----------|--------|-----------|-----|
| **K-means** (recommended) | Automatic clustering | Stable | Most cases |
| **Percentile** | Fixed buckets (top/mid/bottom) | Deterministic | When buckets are clear |
| **Hierarchical** | Agglomerative clustering | Reveals structure | Exploratory, validation |

**Recommendation**: Start with **K-means**, validate with Hierarchical if needed.

#### Step 4: Number of Precision Groups

| Groups | Trade-off | When to Use |
|--------|-----------|------------|
| **3 groups** (simple) | Good accuracy / compression balance | Most cases, fast iteration |
| **4 groups** (fine control) | Finer precision tuning | When 3-group drops are large |

**Recommendation**: Start with 3, bump to 4 if accuracy drop > 2%.

#### Step 5: Bit Allocation

**For 3 groups:**

```
Conservative (recommended):
  Group 0 (high sensitivity):    16-bit   ← Most important layers
  Group 1 (medium sensitivity):   8-bit
  Group 2 (low sensitivity):      4-bit   ← Least important

Aggressive (more compression):
  Group 0:  8-bit
  Group 1:  4-bit
  Group 2:  2-bit
```

**For 4 groups:**

```
Conservative (recommended):
  Group 0:  32-bit  ← Critical layers (usually last 1-2)
  Group 1:  16-bit
  Group 2:   8-bit
  Group 3:   4-bit

Moderate:
  Group 0:  16-bit
  Group 1:   8-bit
  Group 2:   4-bit
  Group 3:   2-bit
```

---

## Configuration Recommendations

### For Non-Fine-Tuned BERT (Pre-trained only)

| Phase | Setting | Value |
|-------|---------|-------|
| Calibration | Size | 5k → final 25k |
| Calibration | Metric | PWCCA (verify with CKA) |
| Clustering | Method | K-means |
| Groups | Count | 3 |
| Bits | Allocation | [16, 8, 4] |
| Evaluation | Test size | 5k → final 25k |

**Expected accuracy drop**: < 2%

### For Fine-Tuned BERT (Like `textattack/bert-base-uncased-imdb`)

| Phase | Setting | Value |
|-------|---------|-------|
| Calibration | Size | 25k (full) |
| Calibration | Metric | PWCCA + CKA |
| Clustering | Method | K-means or Hierarchical |
| Groups | Count | 4 |
| Bits | Allocation | [32, 16, 8, 4] |
| Evaluation | Test size | 25k (full) |

**Expected accuracy drop**: 1-3% with proper tuning

**Special handling:**
- Keep classifier head and embedding in FP32
- Last 2 encoder layers: use 16-bit minimum
- May need QAT (Quantization-Aware Training) if drop > 3%

---

## Output Files

### Phase 1: Sensitivity Files

Location: `Sensitivities/`

```
layer_sensitivity_BERT_IMDB_pwcca.json     # Raw JSON with sensitivity values
layer_sensitivity_BERT_IMDB_pwcca.txt      # Human-readable text format
layer_sensitivity_BERT_IMDB_svcca.json
layer_sensitivity_BERT_IMDB_svcca.txt
layer_sensitivity_BERT_IMDB_cka.json
layer_sensitivity_BERT_IMDB_cka.txt
```

**Format (TXT):**
```
# PWCCA Sensitivities for BERT-base-uncased on IMDB
# Calibration: TRAIN set (25000 examples, Stratified sampling)
model: textattack/bert-base-uncased-imdb
metric: pwcca
calibration_samples: 25000
extraction_time_s: 12.34
num_gpus: 2
batch_size: 64
layer_0	0.123456
layer_1	0.234567
...
layer_11	0.876543
```

### Phase 2: Evaluation Files

Location: `Evaluation/`

```
eval_BERT_IMDB_kmeans_3groups_20250113_142530.txt
eval_BERT_IMDB_percentile_4groups_20250113_150102.txt
eval_BERT_IMDB_hierarchical_3groups_20250113_152344.txt
```

**Format (TXT):**
```
# Mixed-Precision PTQ Results | IMDB | BERT-base-uncased
timestamp: 20250113_142530
model_name: textattack/bert-base-uncased-imdb
device: cuda
num_gpus: 2

# Sensitivity Computation
sensitivity_file: layer_sensitivity_BERT_IMDB_pwcca.json
num_layers: 12

# Evaluation Set
eval_set_size: 25000
batch_size: 32

# Clustering
clustering_method: kmeans
num_clusters: 3

# Bit Allocation
layer_0: 16-bit (sensitivity: 0.123456)
layer_1: 16-bit (sensitivity: 0.234567)
...

# FP32 Baseline (Before Quantization)
accuracy: 0.923450
precision: 0.920123
recall: 0.926789
f1_score: 0.923456

# After Mixed-Precision Quantization
accuracy: 0.921230
precision: 0.918900
recall: 0.924560
f1_score: 0.921728

# Accuracy Drops
accuracy_drop: 0.002220 (0.24%)
precision_drop: 0.001223
recall_drop: 0.002229
f1_drop: 0.001728

# Compression Metrics
orig_bits: 440000000
quant_bits: 220000000
orig_size_mb: 52.38
quant_size_mb: 26.19
compression_ratio: 2.000x
size_reduction_pct: 50.00%
quantization_time_s: 2.345
```

---

## Troubleshooting

### Issue: "Missing sensitivities file"

**Cause**: Haven't run Phase 1 yet, or sensitivity files are in wrong location.

**Solution**:
1. Run Phase 1 first
2. Check that `Sensitivities/` directory exists
3. Verify filenames match expected pattern

### Issue: Large accuracy drop (> 5%)

**Cause**: Bit allocation too aggressive for this model's structure.

**Solutions** (in order):
1. Increase bits for high-sensitivity layers (e.g., 32 instead of 16)
2. Use 4 groups instead of 3
3. Increase calibration size (5k → 25k) for better sensitivities
4. Use PWCCA + CKA together instead of single metric
5. Apply Quantization-Aware Training (QAT) for 1-3 epochs after PTQ

### Issue: Out of memory

**Cause**: Batch size too large or dataset too big.

**Solutions**:
1. Reduce batch size (script auto-selects based on GPU count)
2. Use smaller calibration/evaluation set (5k instead of 25k)
3. Process on CPU (slower but uses RAM) - modify `pick_device()` to return `torch.device("cpu")`

### Issue: Slow extraction of layer outputs

**Cause**: Large dataset or slow GPU.

**Solutions**:
1. Use smaller calibration size (5k instead of 25k) for quick iteration
2. Reduce batch size manually
3. Use fewer workers in DataLoader (modify `num_workers=4`)

---

## Advanced Usage

### Running Specific Phases Only

The script menu allows selecting individual phases:

```bash
# Run only Phase 1
python BERT_base_IMDB.py
# Select "Phase 1: Sensitivity Analysis"

# Run only Phase 2
python BERT_base_IMDB.py
# Select "Phase 2: Quantization & Evaluation"
```

### Custom Bit Allocations

At the "Bit Precision Assignment" step, select "Custom (specify manually)" and enter your own values:
```
e.g., for 3 groups: 32,16,8
e.g., for 4 groups: 16,8,4,2
```

### Comparing Multiple Metrics

Run Phase 1 multiple times with different metrics:
1. First run: Select PWCCA
2. Second run: Select SVCCA (or CKA)
3. Compare results in Phase 2

Then in Phase 2, select different sensitivity files to see impact of each metric.

---

## Performance Benchmarks

### Example Results (Fine-Tuned BERT on IMDB)

| Config | Calib | Metric | Clusters | Bits | FP32 Acc | Quant Acc | Drop | Compression |
|--------|-------|--------|----------|------|----------|-----------|------|-------------|
| Baseline | - | - | - | 32/32/32 | 0.9235 | - | - | 1.0x |
| Conservative | 25k | PWCCA | 3 | 16/8/4 | 0.9235 | 0.9212 | -0.23% | 2.0x |
| Moderate | 25k | PWCCA+CKA | 4 | 16/8/4/2 | 0.9235 | 0.9189 | -0.46% | 2.5x |
| Aggressive | 25k | SVCCA | 4 | 8/4/2/2 | 0.9235 | 0.9045 | -2.05% | 4.0x |

---

## References

### Similarity Metrics

- **PWCCA**: Morcos et al., "Insights on representational similarity in neural networks with canonical correlation" (NeurIPS 2018)
- **SVCCA**: Raghu et al., "SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability" (NeurIPS 2017)
- **CKA**: Kornblith et al., "Similarity of Neural Network Representations Revisited" (ICML 2019)

### Quantization

- **Mixed-Precision Quantization**: Zafrir et al., "Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT"
- **LSQ Quantization**: Zhou et al., "Learned Step Size Quantization" (ICLR 2021)

---

## Citation

If you use this framework, please cite:

```bibtex
@software{bert_quantization_2025,
  title={Mixed-Precision Quantization Framework for BERT on IMDB},
  author={Mixed-Precision Quantization Team},
  year={2025},
  url={https://github.com/amitashnanda/Mixed_Precision_Quantization}
}
```

---

## License

This code is provided as-is for research purposes. See LICENSE for details.

---

## Contact

For issues, questions, or contributions:
- **Repository**: https://github.com/amitashnanda/Mixed_Precision_Quantization
- **Issues**: GitHub Issues page

---

**Last Updated**: January 13, 2025
**Version**: 1.0.0
