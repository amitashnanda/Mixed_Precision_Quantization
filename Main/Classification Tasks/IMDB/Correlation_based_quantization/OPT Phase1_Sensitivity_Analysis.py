"""
Phase 1: Mixed-Precision Quantization - Layer Sensitivity Analysis for OPT Models on IMDB
=========================================================================================

This script computes layer sensitivities using multiple similarity metrics (PWCCA, SVCCA, CKA).
User has complete control over:
1. Model selection (OPT-125M, 350M, 1.3B, 2.7B, 175B)
2. Calibration set size (5k, 10k, 25k)
3. Sampling strategy (random, stratified)
4. Similarity metrics (PWCCA, SVCCA, CKA, or combinations)
5. SVCCA topk parameter

Output: Sensitivity files with metadata about user selections

Usage:
    python Phase1_Sensitivity_Analysis.py

Author: Mixed-Precision Quantization Team
Date: 2025
"""

import os
import json
import time
import random
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

from sklearn.cross_decomposition import CCA
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_DATASETS_CACHE"] = f"{HF_HOME}/datasets"
os.environ["HF_HUB_CACHE"] = f"{HF_HOME}/hub"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)

# OPT Model Configurations
OPT_MODELS = {
    "OPT-125M": {
        "model_name": "facebook/opt-125m",
        "num_layers": 12,
        "hidden_dim": 768,
        "description": "Small - Fast, lightweight (125M parameters)"
    },
    "OPT-350M": {
        "model_name": "facebook/opt-350m",
        "num_layers": 24,
        "hidden_dim": 1024,
        "description": "Small-Medium - Balanced (350M parameters)"
    },
    "OPT-1.3B": {
        "model_name": "facebook/opt-1.3b",
        "num_layers": 24,
        "hidden_dim": 2048,
        "description": "Medium - Good performance (1.3B parameters)"
    },
    "OPT-2.7B": {
        "model_name": "facebook/opt-2.7b",
        "num_layers": 32,
        "hidden_dim": 2560,
        "description": "Medium-Large (2.7B parameters)"
    },
    "OPT-175B": {
        "model_name": "facebook/opt-175b",
        "num_layers": 96,
        "hidden_dim": 12288,
        "description": "Large (175B parameters) - requires multiple GPUs"
    }
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed=42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pick_device():
    """Automatically select the best available device."""
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.device_count()} GPU(s) detected")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print(f"✓ MPS (Apple Silicon) available")
        return torch.device("mps")
    print("⚠ Using CPU (slow)")
    return torch.device("cpu")

def prompt_user(prompt_text, options, default=None):
    """Display a prompt and get user input."""
    print(f"\n{prompt_text}")
    for i, opt in enumerate(options, 1):
        marker = "→ " if opt == default else "  "
        print(f"{marker}{i}. {opt}")
    
    while True:
        try:
            user_input = input(f"Enter choice (1-{len(options)}) [{options.index(default)+1 if default else 1}]: ").strip()
            if not user_input and default:
                return default
            choice_idx = int(user_input) - 1
            if 0 <= choice_idx < len(options):
                return options[choice_idx]
            print(f"❌ Please enter a number between 1 and {len(options)}")
        except ValueError:
            print(f"❌ Invalid input. Please try again.")

def prompt_number(prompt_text, default=None, min_val=None, max_val=None):
    """Get a numeric input from user."""
    print(f"\n{prompt_text}")
    if default:
        print(f"(Default: {default})")
    
    while True:
        try:
            user_input = input("Enter value: ").strip()
            if not user_input and default is not None:
                return default
            value = int(user_input)
            if min_val is not None and value < min_val:
                print(f"❌ Value must be >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"❌ Value must be <= {max_val}")
                continue
            return value
        except ValueError:
            print("❌ Invalid input. Please enter a number.")

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

# ============================================================================
# MODEL SELECTION
# ============================================================================

def select_opt_model():
    """Interactive OPT model selection."""
    print_section("STEP 1: OPT MODEL SELECTION")
    print("""
    Available OPT Models (trained on The Pile):
    """)
    
    model_options = list(OPT_MODELS.keys())
    for key in model_options:
        desc = OPT_MODELS[key]["description"]
        print(f"  • {key}: {desc}")
    
    print("""
    Selection Guide:
    - OPT-125M: Quick testing, limited accuracy
    - OPT-350M: Good balance, fast iteration
    - OPT-1.3B: RECOMMENDED - strong accuracy, manageable size
    - OPT-2.7B: Better performance, more compute
    - OPT-175B: State-of-the-art, requires special setup
    """)
    
    selected_model = prompt_user(
        "Select OPT model:",
        model_options,
        default="OPT-1.3B"
    )
    
    model_config = OPT_MODELS[selected_model]
    print(f"\n✓ Selected: {selected_model}")
    print(f"  HF Hub: {model_config['model_name']}")
    print(f"  Layers: {model_config['num_layers']}")
    print(f"  Hidden Dim: {model_config['hidden_dim']}")
    
    return selected_model, model_config

# ============================================================================
# SIMILARITY METRICS
# ============================================================================

def svcca(X, Y, topk=20, eps=1e-10):
    """Singular Vector Canonical Correlation Analysis."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    
    Ux, Sx, Vxt = np.linalg.svd(Xc, full_matrices=False)
    k_x = min(topk, Sx.size, Xc.shape[1])
    Xr = Ux[:, :k_x] @ np.diag(Sx[:k_x])
    
    Uy, Sy, Vyt = np.linalg.svd(Yc, full_matrices=False)
    k_y = min(topk, Sy.size, Yc.shape[1])
    Yr = Uy[:, :k_y] @ np.diag(Sy[:k_y])
    
    n_comp = min(Xr.shape[1], Yr.shape[1])
    if n_comp <= 0:
        return 0.0
    
    cca = CCA(n_components=n_comp, max_iter=5000)
    try:
        cca.fit(Xr, Yr)
        Xc_c, Yc_c = cca.transform(Xr, Yr)
        
        corrs = []
        for i in range(Xc_c.shape[1]):
            a = Xc_c[:, i]
            b = Yc_c[:, i]
            corr = np.corrcoef(a, b)[0, 1]
            corrs.append(np.nan_to_num(corr))
        
        svcca_val = float(np.mean(corrs))
        return max(min(svcca_val, 1.0), 0.0)
    except Exception:
        return 0.0

def pwcca(X, Y, energy_threshold=0.99, cca_max_components=20, eps=1e-10):
    """Projection-Weighted Canonical Correlation Analysis."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)

    def svd_reduce(A):
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        if S.size == 0:
            return A, np.array([]), np.array([[]])
        energy = np.cumsum(S ** 2) / (np.sum(S ** 2) + eps)
        k = np.searchsorted(energy, energy_threshold) + 1
        k = min(k, S.size)
        return U[:, :k], S[:k], Vt[:k, :]

    Ux, Sx, Vxt = svd_reduce(Xc)
    Uy, Sy, Vyt = svd_reduce(Yc)
    Xr = Xc.dot(Vxt.T) if Vxt.size else Xc
    Yr = Yc.dot(Vyt.T) if Vyt.size else Yc

    n_comp = min(Xr.shape[1], Yr.shape[1], cca_max_components)
    if n_comp <= 0:
        try:
            return float(np.corrcoef(Xc.mean(axis=1), Yc.mean(axis=1))[0, 1])
        except Exception:
            return 0.0

    cca = CCA(n_components=n_comp, max_iter=5000)
    cca.fit(Xr, Yr)
    Xc_c, Yc_c = cca.transform(Xr, Yr)

    corrs = []
    for i in range(Xc_c.shape[1]):
        a = Xc_c[:, i]
        b = Yc_c[:, i]
        denom = (np.std(a) * np.std(b))
        if denom <= eps:
            corr = 0.0
        else:
            corr = float(np.corrcoef(a, b)[0, 1])
        corrs.append(np.nan_to_num(corr))

    corrs = np.array(corrs)

    try:
        w = np.abs(cca.x_weights_).sum(axis=0)
        if np.all(w == 0):
            w = np.ones_like(w)
    except Exception:
        w = np.ones_like(corrs)

    w = w / (w.sum() + eps)
    pwcca_val = float((w * corrs).sum())
    return pwcca_val

def cka(X, Y, debiased=True):
    """Centered Kernel Alignment (CKA)."""
    def centering(K):
        n = K.shape[0]
        unit = np.ones([n, n]) / n
        I = np.eye(n)
        H = I - unit
        return np.dot(np.dot(H, K), H)

    def linear_hsic(X, Y):
        KX = np.dot(X, X.T)
        KY = np.dot(Y, Y.T)
        KX_centered = centering(KX)
        KY_centered = centering(KY)
        return np.trace(np.dot(KX_centered, KY_centered))

    hsic_xy = linear_hsic(X, Y)
    hsic_xx = linear_hsic(X, X)
    hsic_yy = linear_hsic(Y, Y)

    if hsic_xx <= 0 or hsic_yy <= 0:
        return 0.0

    if debiased:
        cka_val = hsic_xy / np.sqrt(hsic_xx * hsic_yy)
    else:
        cka_val = hsic_xy / (np.sqrt(hsic_xx) * np.sqrt(hsic_yy) + 1e-10)

    return float(max(min(cka_val, 1.0), 0.0))

# ============================================================================
# DATA & FEATURE EXTRACTION
# ============================================================================

@torch.no_grad()
def extract_layer_outputs(dataloader, model, device, num_layers=None):
    """Extract hidden states from all layers."""
    model.eval()
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    actual_model = model.module if hasattr(model, 'module') else model
    if num_layers is None:
        num_layers = actual_model.config.num_hidden_layers + 1
    
    layer_outputs = {f"layer_{i}": [] for i in range(num_layers)}

    for batch in tqdm(dataloader, desc="Extracting layer outputs"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = out.hidden_states
        
        for i, hs in enumerate(hidden_states):
            mask = attention_mask.unsqueeze(-1).to(hs.dtype)
            summed = (hs * mask).sum(dim=1)
            lens = mask.sum(dim=1).clamp(min=1.0)
            mean_pooled = (summed / lens).cpu().numpy()
            layer_outputs[f"layer_{i}"].append(mean_pooled)
    
    for k in layer_outputs:
        layer_outputs[k] = np.vstack(layer_outputs[k])
    
    return layer_outputs

# ============================================================================
# SENSITIVITY COMPUTATION
# ============================================================================

def compute_layer_sensitivity_pwcca(layer_outputs, target_layer_index):
    """Compute sensitivity using PWCCA metric."""
    target_key = f"layer_{target_layer_index}"
    target = layer_outputs[target_key]
    corrs = []
    for key, data in layer_outputs.items():
        if key == target_key:
            continue
        n = min(len(target), len(data))
        sim = pwcca(target[:n], data[:n])
        sim = max(min(sim, 1.0), -1.0)
        corrs.append(sim)
    mean_corr = float(np.mean(corrs)) if corrs else 0.0
    return target_key, 1.0 - mean_corr

def compute_layer_sensitivity_svcca(layer_outputs, target_layer_index, topk=20):
    """Compute sensitivity using SVCCA metric."""
    target_key = f"layer_{target_layer_index}"
    target = layer_outputs[target_key]
    corrs = []
    for key, data in layer_outputs.items():
        if key == target_key:
            continue
        n = min(len(target), len(data))
        sim = svcca(target[:n], data[:n], topk=topk)
        sim = max(min(sim, 1.0), 0.0)
        corrs.append(sim)
    mean_corr = float(np.mean(corrs)) if corrs else 0.0
    return target_key, 1.0 - mean_corr

def compute_layer_sensitivity_cka(layer_outputs, target_layer_index):
    """Compute sensitivity using CKA metric."""
    target_key = f"layer_{target_layer_index}"
    target = layer_outputs[target_key]
    corrs = []
    for key, data in layer_outputs.items():
        if key == target_key:
            continue
        n = min(len(target), len(data))
        sim = cka(target[:n], data[:n])
        sim = max(min(sim, 1.0), 0.0)
        corrs.append(sim)
    mean_corr = float(np.mean(corrs)) if corrs else 0.0
    return target_key, 1.0 - mean_corr

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main Phase 1 sensitivity analysis pipeline."""
    print_section("PHASE 1: LAYER SENSITIVITY ANALYSIS FOR OPT MODELS ON IMDB")
    print("""
    This phase computes layer sensitivities using selected similarity metrics.
    
    Complete user control over:
    ✓ Model selection (5 OPT variants)
    ✓ Calibration set size (5k, 10k, 25k)
    ✓ Sampling strategy (random, stratified)
    ✓ Similarity metrics (PWCCA, SVCCA, CKA, or combinations)
    ✓ SVCCA topk parameter
    
    Output: Sensitivity files with complete metadata
    """)
    
    set_seed(42)
    device = pick_device()
    
    # ========== STEP 1: Model Selection ==========
    selected_model, model_config = select_opt_model()
    model_name = model_config["model_name"]
    num_layers = model_config["num_layers"]
    
    # ========== STEP 2: Calibration Size ==========
    print_section("STEP 2: CALIBRATION SET SIZE")
    print("""
    Calibration data is used to extract layer activations for sensitivity computation.
    
    Options:
    - 5k: Fast (2-3 min), noisier estimates - good for quick testing
    - 10k: Medium (5-8 min), balanced - development iterations
    - 25k: Full IMDB train (10-15 min), most stable - RECOMMENDED for final results
    
    Tip: Start with 5k for fast iteration, use 25k for publication results.
    """)
    
    calib_options = ["5k (fast, noisy)", "10k (medium)", "25k (full IMDB train, recommended)"]
    calib_choice = prompt_user("Select calibration size:", calib_options, default="25k (full IMDB train, recommended)")
    
    calib_size_map = {
        "5k (fast, noisy)": 5000,
        "10k (medium)": 10000,
        "25k (full IMDB train, recommended)": 25000
    }
    calib_size = calib_size_map[calib_choice]
    
    # ========== STEP 3: Load & Prepare Calibration Data ==========
    print(f"\n📥 Loading IMDB dataset (TRAIN split for calibration)...")
    ds = load_dataset("imdb")
    calibration_set = ds["train"].select(range(min(calib_size, len(ds["train"]))))
    print(f"✓ Loaded {len(calibration_set)} examples for calibration")
    
    # ========== STEP 4: Sampling Strategy ==========
    print_section("STEP 3: SAMPLING STRATEGY")
    print("""
    How to sample from calibration data:
    
    - Random: Simple random sampling
    - Stratified: Balanced by label (50/50 pos/neg) - RECOMMENDED
      → Ensures equal representation of both sentiment classes
    """)
    
    sampling_choice = prompt_user(
        "Select sampling strategy:",
        ["Random sampling", "Stratified sampling (recommended)"],
        default="Stratified sampling (recommended)"
    )
    
    if "Stratified" in sampling_choice:
        pos_examples = [ex for ex in calibration_set if ex["label"] == 1]
        neg_examples = [ex for ex in calibration_set if ex["label"] == 0]
        
        n_take = min(len(pos_examples), len(neg_examples), calib_size // 2)
        calibration_set = calibration_set.__class__(
            pos_examples[:n_take] + neg_examples[:n_take]
        )
        print(f"✓ Using stratified sampling: {len(calibration_set)} examples (50/50 pos/neg split)")
    else:
        print(f"✓ Using random sampling: {len(calibration_set)} examples")
    
    # ========== STEP 5: Tokenize ==========
    print_section("STEP 4: TOKENIZING DATA")
    print(f"Tokenizing {len(calibration_set)} examples with max_length=512...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    
    calibration_tokenized = calibration_set.map(tokenize_fn, batched=True)
    calibration_tokenized = calibration_tokenized.remove_columns(
        [c for c in calibration_tokenized.column_names if c not in ["input_ids", "attention_mask", "label"]]
    )
    calibration_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    batch_size = 64 if torch.cuda.device_count() > 1 else 16
    loader = DataLoader(
        calibration_tokenized, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    print(f"✓ Tokenization complete. Batch size: {batch_size}")
    
    # ========== STEP 6: Load Model ==========
    print_section("STEP 5: LOADING MODEL")
    print(f"Loading {selected_model} from HuggingFace Hub...")
    base_model = AutoModel.from_pretrained(model_name).to(device)
    print(f"✓ Model loaded successfully")
    
    # ========== STEP 7: Similarity Metric Selection ==========
    print_section("STEP 6: SIMILARITY METRIC SELECTION")
    print("""
    Choose how to measure representational similarity between layers:
    
    PWCCA (Projection-Weighted CCA):
    - Projects activations onto energy-significant subspaces
    - Computes weighted canonical correlation
    - Pros: Robust, interpretable, good sensitivity estimates
    - Cons: Slightly slower than SVCCA
    
    SVCCA (SVD + CCA):
    - Faster alternative using top singular vectors
    - Controlled by 'topk' parameter (default: 20)
    - Pros: Fast, lower memory footprint
    - Cons: Less stable on small calibration sets
    
    CKA (Centered Kernel Alignment):
    - Kernel-based similarity without dimension reduction
    - Pros: No hyperparameters, robust to noise
    - Cons: O(n²) memory for large batches
    
    Recommendation: Use PWCCA + CKA together (complementary views)
    Alternative: Use SVCCA for faster results
    """)
    
    metric_options = [
        "PWCCA only",
        "SVCCA only",
        "CKA only",
        "PWCCA + CKA (recommended)",
        "PWCCA + SVCCA",
        "SVCCA + CKA",
        "All three (PWCCA + SVCCA + CKA)"
    ]
    metric_choice = prompt_user(
        "Select similarity metric(s):",
        metric_options,
        default="PWCCA + CKA (recommended)"
    )
    
    metrics_to_use = []
    svcca_topk = None
    
    if "PWCCA" in metric_choice:
        metrics_to_use.append("pwcca")
    if "SVCCA" in metric_choice:
        metrics_to_use.append("svcca")
    if "CKA" in metric_choice:
        metrics_to_use.append("cka")
    
    # ========== STEP 8: SVCCA topk parameter ==========
    if "svcca" in metrics_to_use:
        print_section("STEP 7: SVCCA CONFIGURATION")
        print("""
        SVCCA uses SVD to reduce dimensionality before CCA.
        
        topk parameter: Number of top singular vectors to keep
        - Default: 20 (good balance)
        - Larger values: More information preserved, slower
        - Smaller values: Faster, more aggressive compression
        
        Recommendation: Keep default (20) unless you have specific needs
        """)
        svcca_topk = prompt_number(
            "Enter SVCCA topk parameter:",
            default=20,
            min_val=1,
            max_val=64
        )
    
    # ========== STEP 9: Extract Layer Outputs ==========
    print_section("STEP 8: EXTRACTING LAYER OUTPUTS")
    print(f"Extracting hidden states from {num_layers + 1} layers...")
    print(f"Processing {len(calibration_set)} examples...")
    
    t0 = time.time()
    layer_outputs = extract_layer_outputs(loader, base_model, device, num_layers=num_layers + 1)
    extract_time = time.time() - t0
    print(f"✓ Extracted {len(layer_outputs)} layers in {extract_time:.2f}s")
    
    # ========== STEP 10: Compute Sensitivities ==========
    print_section("STEP 9: COMPUTING LAYER SENSITIVITIES")
    
    sensitivities_dict = {}
    computation_times = {}
    
    for metric in metrics_to_use:
        print(f"\n▶ Computing {metric.upper()}-based Sensitivities...")
        sensitivities = {}
        
        metric_t0 = time.time()
        
        if metric == "pwcca":
            for i in tqdm(range(1, num_layers + 1), desc=f"PWCCA"):
                key, sens = compute_layer_sensitivity_pwcca(layer_outputs, i)
                enc_idx = i - 1
                sensitivities[f"layer_{enc_idx}"] = float(sens)
        
        elif metric == "svcca":
            for i in tqdm(range(1, num_layers + 1), desc=f"SVCCA (topk={svcca_topk})"):
                key, sens = compute_layer_sensitivity_svcca(layer_outputs, i, topk=svcca_topk)
                enc_idx = i - 1
                sensitivities[f"layer_{enc_idx}"] = float(sens)
        
        elif metric == "cka":
            for i in tqdm(range(1, num_layers + 1), desc=f"CKA"):
                key, sens = compute_layer_sensitivity_cka(layer_outputs, i)
                enc_idx = i - 1
                sensitivities[f"layer_{enc_idx}"] = float(sens)
        
        metric_t1 = time.time()
        computation_times[metric] = metric_t1 - metric_t0
        sensitivities_dict[metric] = sensitivities
        
        # Print sensitivities
        print(f"\n{metric.upper()} Sensitivities:")
        print(f"{'Layer':<10} {'Sensitivity':<15} {'Interpretation':<30}")
        print("-" * 55)
        for layer_name in sorted(sensitivities, key=lambda x: int(x.split("_")[1])):
            sens_val = sensitivities[layer_name]
            if sens_val < 0.3:
                interp = "Low (can use 2-4 bits)"
            elif sens_val < 0.6:
                interp = "Medium (use 8 bits)"
            else:
                interp = "High (use 16+ bits)"
            print(f"{layer_name:<10} {sens_val:<15.6f} {interp:<30}")
    
    # ========== STEP 11: Save Results ==========
    print_section("STEP 10: SAVING SENSITIVITY FILES")
    
    os.makedirs("Sensitivities", exist_ok=True)
    saved_files = []
    
    for metric, sensitivities in sensitivities_dict.items():
        # Create descriptive filename
        calib_label = calib_choice.split()[0]  # e.g., "5k", "10k", "25k"
        sampling_label = "stratified" if "Stratified" in sampling_choice else "random"
        svcca_suffix = f"_topk{svcca_topk}" if metric == "svcca" else ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        json_filename = f"sens_{selected_model}_{metric}_calib{calib_label}_{sampling_label}{svcca_suffix}_{timestamp}.json"
        json_path = os.path.join("Sensitivities", json_filename)
        
        # Save JSON
        with open(json_path, "w") as f:
            json.dump(sensitivities, f, indent=2)
        saved_files.append(json_filename)
        
        # Save TXT with metadata
        txt_filename = f"sens_{selected_model}_{metric}_calib{calib_label}_{sampling_label}{svcca_suffix}_{timestamp}.txt"
        txt_path = os.path.join("Sensitivities", txt_filename)
        
        with open(txt_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write(f"SENSITIVITY ANALYSIS RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("="*80 + "\n")
            f.write("USER CONFIGURATIONS\n")
            f.write("="*80 + "\n")
            f.write(f"Model: {selected_model}\n")
            f.write(f"Model HF Hub: {model_name}\n")
            f.write(f"Num Layers: {num_layers}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            
            f.write(f"Calibration Set Size: {calib_choice}\n")
            f.write(f"Calibration Samples: {len(calibration_set)}\n")
            f.write(f"Sampling Strategy: {sampling_choice}\n")
            f.write(f"Similarity Metric: {metric.upper()}\n")
            if metric == "svcca":
                f.write(f"SVCCA topk: {svcca_topk}\n")
            f.write(f"\n")
            
            f.write("="*80 + "\n")
            f.write("COMPUTATION DETAILS\n")
            f.write("="*80 + "\n")
            f.write(f"Layer Extraction Time: {extract_time:.2f}s\n")
            f.write(f"Sensitivity Computation Time ({metric.upper()}): {computation_times[metric]:.2f}s\n")
            f.write(f"Total Phase 1 Time: {extract_time + computation_times[metric]:.2f}s\n")
            f.write(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
            f.write(f"Num GPUs: {torch.cuda.device_count()}\n")
            f.write(f"Batch Size: {batch_size}\n\n")
            
            f.write("="*80 + "\n")
            f.write(f"LAYER SENSITIVITIES ({metric.upper()})\n")
            f.write("="*80 + "\n")
            f.write(f"{'Layer':<12} {'Sensitivity':<15} {'Status':<30}\n")
            f.write("-" * 80 + "\n")
            
            for layer_name in sorted(sensitivities, key=lambda x: int(x.split("_")[1])):
                sens_val = sensitivities[layer_name]
                if sens_val < 0.3:
                    status = "LOW (2-4 bits OK)"
                elif sens_val < 0.6:
                    status = "MEDIUM (8 bits recommended)"
                else:
                    status = "HIGH (16+ bits needed)"
                f.write(f"{layer_name:<12} {sens_val:<15.6f} {status:<30}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("="*80 + "\n")
            sens_array = np.array(list(sensitivities.values()))
            f.write(f"Mean Sensitivity: {sens_array.mean():.6f}\n")
            f.write(f"Min Sensitivity:  {sens_array.min():.6f}\n")
            f.write(f"Max Sensitivity:  {sens_array.max():.6f}\n")
            f.write(f"Std Deviation:    {sens_array.std():.6f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("INTERPRETATION GUIDE\n")
            f.write("="*80 + "\n")
            f.write("""
Layer Sensitivity indicates how much that layer affects model output.

HIGH sensitivity (0.7-1.0):
  → Keep at high precision (16-32 bits)
  → Usually top layers with task-specific information
  
MEDIUM sensitivity (0.4-0.7):
  → Use standard precision (8 bits)
  → Middle layers with general features
  
LOW sensitivity (0.0-0.4):
  → Can use aggressive compression (2-4 bits)
  → Early layers with redundant features
  
Note: High sensitivity layers typically at the output end of the model.
""")
        
        print(f"✓ Saved {metric.upper()} sensitivities to:")
        print(f"  JSON: {json_filename}")
        print(f"  TXT:  {txt_filename}")
    
    # ========== FINAL SUMMARY ==========
    print_section("PHASE 1 COMPLETE - SENSITIVITY ANALYSIS RESULTS")
    print(f"""
✓ Successfully computed layer sensitivities for {selected_model}

Saved Files ({len(saved_files)} files):
""")
    for i, filename in enumerate(saved_files, 1):
        print(f"  {i}. {filename}")
    
    print(f"""
Next Steps:
1. Review sensitivity files in the 'Sensitivities/' folder
2. Understand which layers are most/least sensitive
3. Run Phase 2 using these sensitivity files for quantization
4. Choose bit-widths based on layer sensitivities

File Naming Format:
  sens_<MODEL>_<METRIC>_calib<SIZE>_<SAMPLING>_<TIMESTAMP>

Example: sens_OPT-1.3B_pwcca_calib25k_stratified_20250118_143022

This makes it easy to identify and match sensitivity files in Phase 2.
""")

if __name__ == "__main__":
    main()
