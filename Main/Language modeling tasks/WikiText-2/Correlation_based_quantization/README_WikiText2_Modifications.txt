# Modified Phase 1 & Phase 2 for WikiText-2 Language Modeling

## Files Created

### 1. **OPT_Phase1_WikiText_Sensitivity_Analysis.py**
Modified Phase 1 for WikiText-2 dataset (Language Modeling - PTQ)

**Key Changes from IMDB version:**
- ✅ Uses **WikiText-2 dataset** instead of IMDB
- ✅ Calibration size options: **128, 256, 512, 1k, 5k, 10k** (industry-standard PTQ sizes)
- ✅ Random sampling (no stratification needed for language modeling)
- ✅ Ensures tokenizer has `pad_token = eos_token` for OPT
- ✅ No label column (language modeling task, not classification)
- ✅ Outputs sensitivity files with WikiText-2 specific metadata

**Workflow:**
```
Step 0: Select OPT model (125M, 350M, 1.3B, 2.7B, 175B)
Step 1: Load OPT model
Step 2: Select calibration size (128-10k samples for PTQ)
Step 3: Load WikiText-2 training data
Step 4: Tokenize (max_length=512)
Step 5: Select similarity metrics (PWCCA, SVCCA, CKA)
Step 6: Extract layer outputs
Step 7: Compute sensitivities
Step 8: Save to Sensitivities/ folder
```

**Output Files:**
- `Sensitivities/sens_OPT-XXX_WikiText2_XXXX_metric_timestamp.json`
- `Sensitivities/sens_OPT-XXX_WikiText2_XXXX_metric_timestamp.txt` (with metadata)

---

### 2. **OPT_Phase2_WikiText_Training_Evaluation.py**
Modified Phase 2 for WikiText-2 training (Language Modeling)

**Key Changes from IMDB version:**
- ✅ Uses **WikiText-2 training set** instead of test evaluation
- ✅ **Training on quantized model** (not just evaluation)
- ✅ Language modeling evaluation metric: **Perplexity** (not accuracy)
- ✅ Custom LMHead for language modeling task
- ✅ Training loop with gradient updates on WikiText-2 training data
- ✅ Validation on WikiText-2 validation set
- ✅ User control over training parameters (batch size, learning rate, epochs)

**Workflow:**
```
Step 1: Load sensitivity file from Phase 1
Step 2: Load OPT model and add LM head
Step 3: Load WikiText-2 (train, validation, test splits)
Step 4: Configure training hyperparameters
Step 5: Select clustering strategy (K-means, Percentile, Hierarchical)
Step 6: Select number of precision groups (3 or 4)
Step 7: Perform layer clustering
Step 8: Select bit allocation
Step 9: Apply mixed-precision quantization
Step 10: Evaluate baseline (before training)
Step 11: Train quantized model for N epochs
Step 12: Final validation and results
Step 13: Save training log
```

**Output Files:**
- `Evaluation/train_OPT-XXX_WikiText2_kmeans_3g_timestamp.txt` (training log)

**Saved Metrics:**
- Baseline perplexity (pre-training)
- Train loss and perplexity per epoch
- Validation perplexity per epoch
- Final perplexity
- Model compression statistics

---

## Key Differences from IMDB Versions

| Aspect | IMDB Phase 1/2 | WikiText Phase 1/2 |
|--------|---|---|
| **Dataset** | IMDB (classification) | WikiText-2 (language modeling) |
| **Task** | Sentiment classification | Language modeling (Causal LM) |
| **Calibration** | 5k, 10k, 25k | **128, 256, 512, 1k, 5k, 10k** (PTQ) |
| **Sampling** | Stratified (pos/neg) | Random (no labels) |
| **Evaluation Metric** | Accuracy, Precision, Recall, F1 | **Perplexity** |
| **Phase 2 Task** | Evaluation only | **Training + Evaluation** |
| **Training** | No | **Yes (configurable epochs)** |
| **Labels** | Yes | No |
| **Model Head** | Classification head | **Language Modeling head** |

---

## How to Run

### Phase 1: Sensitivity Analysis
```bash
cd "Main/Language modeling tasks/WikiText-2/Correlation_based_quantization"
python OPT_Phase1_WikiText_Sensitivity_Analysis.py
```

**Interactive prompts:**
1. Select OPT model (default: OPT-1.3B)
2. Select calibration size (default: 1k - **recommended for PTQ**)
3. Select similarity metric(s) (default: PWCCA + CKA)

**Output:** Sensitivity files in `Sensitivities/` folder

---

### Phase 2: Quantization & Training
```bash
python OPT_Phase2_WikiText_Training_Evaluation.py
```

**Interactive prompts:**
1. Select sensitivity file from Phase 1
2. Select batch size (default: 16)
3. Select learning rate (default: 0.0001)
4. Select number of epochs (default: 3)
5. Select clustering strategy (default: K-means)
6. Select number of groups (default: 3)
7. Select bit allocation (default: [16, 8, 4])

**Output:** Training log in `Evaluation/` folder

---

## Recommended Settings for Quick Testing

### Phase 1:
- OPT-125M (fastest)
- 256 calibration samples (industry standard, ~1-2 min)
- PWCCA + CKA metrics

### Phase 2:
- Batch size: 16
- Learning rate: 0.0001
- Epochs: 1 (quick test)
- Clustering: K-means
- Groups: 3
- Bits: [16, 8, 4]

**Total time:** ~10-15 minutes

---

## Recommended Settings for Production

### Phase 1:
- OPT-1.3B (strong balance)
- 1k calibration samples (~3-5 min)
- PWCCA + CKA metrics

### Phase 2:
- Batch size: 32
- Learning rate: 0.0001
- Epochs: 3
- Clustering: K-means
- Groups: 3 or 4
- Bits: [32, 16, 8, 4] (4 groups) or [16, 8, 4] (3 groups)

**Total time:** ~30-45 minutes

---

## Key Features

✅ **Complete User Control:** Every step interactive with defaults
✅ **PTQ Industry Standard:** Uses 128-1k calibration samples (standard practice)
✅ **Multiple Metrics:** PWCCA, SVCCA, CKA for robust analysis
✅ **Flexible Clustering:** K-means, Percentile, Hierarchical
✅ **Mixed-Precision:** Support for 2-32 bit per layer
✅ **Language Modeling:** Proper causal LM setup for WikiText-2
✅ **Training Support:** Phase 2 trains on WikiText-2 train set
✅ **Perplexity Metric:** Standard LM evaluation metric
✅ **GPU Optimized:** Multi-GPU support, proper memory management
✅ **Comprehensive Logging:** All configurations saved in output files

---

## Important Notes

1. **WikiText-2 Structure:**
   - Train: For training quantized models
   - Validation: For validation during training
   - Test: Available but not used in Phase 2

2. **PTQ Calibration:**
   - 256-1k samples is industry standard
   - More samples = more stable but slower
   - 128 is ultra-fast for quick iteration

3. **Training:**
   - Phase 2 trains the quantized model
   - Gradient updates help recover accuracy
   - Batch size and learning rate can be tuned per run

4. **File Naming:**
   - `sen s_OPT-1.3B_WikiText2_1000samples_pwcca_20250116_143022.json`
   - `train_OPT-1.3B_WikiText2_kmeans_3g_20250116_143022.txt`

---

## Troubleshooting

**OOM (Out of Memory):**
- Reduce batch size (use 8 instead of 16)
- Use smaller model (OPT-125M instead of 1.3B)
- Reduce calibration size to 256

**Slow Training:**
- Reduce number of epochs
- Reduce batch size (faster iterations)
- Use smaller model

**High Perplexity Drop:**
- Use more aggressive bit allocation (e.g., [32/16/8/4] instead of [16/8/4])
- Increase training epochs
- Use higher learning rate

---

Created: November 2025
Modified from IMDB classification to WikiText-2 language modeling
