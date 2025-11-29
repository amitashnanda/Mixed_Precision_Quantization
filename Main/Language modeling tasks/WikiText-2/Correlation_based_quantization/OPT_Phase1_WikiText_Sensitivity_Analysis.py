"""
Phase 1: Sensitivity Analysis for OPT Models on WikiText-2 (Language Modeling Task - PTQ)
=========================================================================================

This script computes layer sensitivities using:
1. Model selection (OPT-125M, OPT-350M, OPT-1.3B, OPT-2.7B, OPT-175B)
2. Calibration set size selection for PTQ (128, 256, 512, 1k, 5k, 10k)
3. Similarity metric selection (PWCCA, SVCCA, CKA)
4. WikiText-2 dataset for language modeling task
5. Layer output extraction
6. Sensitivity computation

User has complete control over all parameters.

Output: Sensitivity JSON and TXT files with metadata (saved in Sensitivities/ folder)

Dataset: WikiText-2 (https://huggingface.co/datasets/Salesforce/wikitext)
Task: Language Modeling (Causal LM)

Usage:
    python OPT_Phase1_WikiText_Sensitivity_Analysis.py

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
from sklearn.cluster import KMeans

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
    print_section("STEP 0: OPT MODEL SELECTION")
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
    print_section("PHASE 1: SENSITIVITY ANALYSIS FOR OPT MODELS ON WIKITEXT-2")
    print("""
    This phase computes layer sensitivities for Language Modeling (PTQ) using:
    ✓ WikiText-2 dataset for calibration
    ✓ Multiple similarity metrics (PWCCA, SVCCA, CKA)
    ✓ Industry-standard PTQ calibration sizes (128-10k samples)
    
    Complete user control over:
    ✓ OPT model selection
    ✓ Calibration set size
    ✓ Similarity metric(s)
    
    Output: Sensitivity files saved in Sensitivities/ folder
    """)
    
    set_seed(42)
    device = pick_device()
    
    # ========== STEP 0: Model Selection ==========
    selected_model, model_config = select_opt_model()
    model_name = model_config["model_name"]
    num_layers = model_config["num_layers"]
    
    # ========== STEP 1: Load Model ==========
    print_section("STEP 1: LOADING OPT MODEL")
    print(f"Loading {selected_model} from HuggingFace Hub...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    base_model = AutoModel.from_pretrained(model_name).to(device)
    print(f"✓ Model loaded: {num_layers} layers")
    
    # ========== STEP 2: Calibration Size ==========
    print_section("STEP 2: CALIBRATION SET SIZE (PTQ)")
    print("""
    Calibration data extracts layer activations for sensitivity computation.
    
    For Post-Training Quantization (PTQ) on WikiText-2:
    
    Ultra-Fast PTQ:
    - 128 samples: Very fast (30-60s), minimal memory, quick trends
    - 256 samples: Fast (1-2 min), good for iteration, INDUSTRY STANDARD
    
    Balanced PTQ:
    - 512 samples: Medium (2-3 min), stable estimates
    - 1k samples: Standard (3-5 min), reliable sensitivities, RECOMMENDED
    
    Stable PTQ:
    - 5k samples: Slower (8-10 min), high-quality estimates
    - 10k samples: Very stable (12-15 min), robust sensitivities
    
    Recommendation: Use 256-1k for most PTQ applications
    """)
    
    calib_options = [
        "128 (ultra-fast PTQ, industry standard)",
        "256 (fast PTQ, commonly used)",
        "512 (balanced PTQ)",
        "1k (stable PTQ, recommended)",
        "5k (higher quality)",
        "10k (very stable)"
    ]
    calib_choice = prompt_user(
        "Select calibration size for PTQ:",
        calib_options,
        default="1k (stable PTQ, recommended)"
    )
    
    calib_size_map = {
        "128 (ultra-fast PTQ, industry standard)": 128,
        "256 (fast PTQ, commonly used)": 256,
        "512 (balanced PTQ)": 512,
        "1k (stable PTQ, recommended)": 1000,
        "5k (higher quality)": 5000,
        "10k (very stable)": 10000
    }
    calib_size = calib_size_map[calib_choice]
    
    # ========== STEP 3: Load WikiText-2 ==========
    print_section("STEP 3: LOADING WIKITEXT-2 DATASET")
    print(f"Loading WikiText-2 for language modeling task...")
    
    try:
        ds = load_dataset("wikitext", "wikitext-2-v1")
    except:
        print("Note: Using wikitext-2 as fallback...")
        ds = load_dataset("wikitext", "wikitext-2")
    
    # Filter out empty examples
    train_data = ds["train"]
    non_empty = [ex for ex in train_data if ex.get("text", "").strip()]
    print(f"Total non-empty train examples: {len(non_empty)}")
    
    # Random sampling for calibration
    indices = np.random.choice(len(non_empty), size=min(calib_size, len(non_empty)), replace=False)
    calibration_set = train_data.select(indices.tolist())
    print(f"✓ Randomly sampled {len(calibration_set)} examples for calibration")
    
    # ========== STEP 4: Tokenize ==========
    print_section("STEP 4: TOKENIZING DATA")
    print(f"Tokenizing {len(calibration_set)} WikiText-2 examples...")
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    
    calibration_tokenized = calibration_set.map(tokenize_fn, batched=True)
    calibration_tokenized = calibration_tokenized.remove_columns(
        [c for c in calibration_tokenized.column_names if c not in ["input_ids", "attention_mask"]]
    )
    calibration_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # Smart batch size
    if calib_size <= 256:
        batch_size = 16
    elif calib_size <= 512:
        batch_size = 32
    else:
        batch_size = 64 if torch.cuda.device_count() > 1 else 32
    
    loader = DataLoader(
        calibration_tokenized, batch_size=batch_size, shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0, pin_memory=True
    )
    print(f"✓ Tokenization complete. Batch size: {batch_size}")
    
    # ========== STEP 5: Similarity Metrics ==========
    print_section("STEP 5: SIMILARITY METRIC SELECTION")
    print("""
    How to measure representational similarity between layers:
    
    PWCCA (Projection-Weighted CCA):
    - Projects onto energy-significant subspaces
    - Pros: Robust, interpretable, good for sensitivity
    - Cons: Slightly slower than SVCCA
    - Use when: You want reliable sensitivity estimates
    
    SVCCA (SVD + CCA):
    - Faster alternative to PWCCA
    - Controlled via topk parameter (default: 20)
    - Pros: Fast, lower memory
    - Cons: Less stable on small calibration sets
    
    CKA (Centered Kernel Alignment):
    - Kernel-based similarity without dimension reduction
    - Pros: No hyperparameters, robust to noise
    - Cons: O(n²) memory for large batches
    
    Recommendation: PWCCA + CKA (complementary views)
    """)
    
    metric_options = [
        "PWCCA only",
        "SVCCA only",
        "CKA only",
        "PWCCA + CKA (recommended)",
        "All three (PWCCA + SVCCA + CKA)"
    ]
    metric_choice = prompt_user(
        "Select similarity metric(s):",
        metric_options,
        default="PWCCA + CKA (recommended)"
    )
    
    metrics_to_use = []
    if "PWCCA" in metric_choice:
        metrics_to_use.append("pwcca")
    if "SVCCA" in metric_choice:
        metrics_to_use.append("svcca")
    if "CKA" in metric_choice:
        metrics_to_use.append("cka")
    
    # ========== STEP 6: Extract Layer Outputs ==========
    print_section("STEP 6: EXTRACTING LAYER OUTPUTS")
    print(f"Processing {len(calibration_set)} calibration examples...")
    t0 = time.time()
    layer_outputs = extract_layer_outputs(loader, base_model, device, num_layers=num_layers + 1)
    extract_time = time.time() - t0
    print(f"✓ Extracted {len(layer_outputs)} layers in {extract_time:.2f}s")
    
    # ========== STEP 7: Compute Sensitivities ==========
    sensitivities_dict = {}
    
    for metric in metrics_to_use:
        print_section(f"STEP 7.{metrics_to_use.index(metric)+1}: COMPUTING {metric.upper()}-BASED SENSITIVITIES")
        sensitivities = {}
        
        if metric == "pwcca":
            for i in tqdm(range(1, num_layers + 1), desc="PWCCA sensitivity"):
                key, sens = compute_layer_sensitivity_pwcca(layer_outputs, i)
                enc_idx = i - 1
                sensitivities[f"layer_{enc_idx}"] = float(sens)
        
        elif metric == "svcca":
            svcca_topk = prompt_number(
                "Enter SVCCA topk parameter (number of top singular vectors):",
                default=20,
                min_val=1,
                max_val=64
            )
            for i in tqdm(range(1, num_layers + 1), desc="SVCCA sensitivity"):
                key, sens = compute_layer_sensitivity_svcca(layer_outputs, i, topk=svcca_topk)
                enc_idx = i - 1
                sensitivities[f"layer_{enc_idx}"] = float(sens)
        
        elif metric == "cka":
            for i in tqdm(range(1, num_layers + 1), desc="CKA sensitivity"):
                key, sens = compute_layer_sensitivity_cka(layer_outputs, i)
                enc_idx = i - 1
                sensitivities[f"layer_{enc_idx}"] = float(sens)
        
        sensitivities_dict[metric] = sensitivities
        
        # Print sensitivities
        print(f"\n{metric.upper()} Sensitivities:")
        for layer_name in sorted(sensitivities, key=lambda x: int(x.split("_")[1])):
            print(f"  {layer_name}: {sensitivities[layer_name]:.6f}")
    
    # ========== STEP 8: Save Results ==========
    print_section("STEP 8: SAVING SENSITIVITY FILES")
    
    os.makedirs("Sensitivities", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for metric, sensitivities in sensitivities_dict.items():
        # Generate informative filename
        filename_base = f"sens_{selected_model}_WikiText2_{calib_size}samples_{metric}_{timestamp}"
        
        # Save JSON
        json_filename = f"{filename_base}.json"
        json_path = os.path.join("Sensitivities", json_filename)
        with open(json_path, "w") as f:
            json.dump(sensitivities, f, indent=2)
        
        # Save TXT with metadata
        txt_filename = f"{filename_base}.txt"
        txt_path = os.path.join("Sensitivities", txt_filename)
        with open(txt_path, "w") as f:
            f.write("="*80 + "\n")
            f.write("LAYER SENSITIVITY FILE - LANGUAGE MODELING (PTQ) ON WIKITEXT-2\n")
            f.write("="*80 + "\n\n")
            
            f.write("="*80 + "\n")
            f.write("CONFIGURATION\n")
            f.write("="*80 + "\n")
            f.write(f"Model: {selected_model}\n")
            f.write(f"Model HF Hub: {model_name}\n")
            f.write(f"Task: Language Modeling (Causal LM)\n")
            f.write(f"Dataset: WikiText-2\n")
            f.write(f"Calibration Set Size: {calib_size} samples\n")
            f.write(f"Sampling Strategy: Random sampling\n")
            f.write(f"Similarity Metric: {metric.upper()}\n")
            f.write(f"Max Sequence Length: 512 tokens\n")
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Num Layers: {num_layers}\n")
            f.write(f"Hidden Dimension: {model_config['hidden_dim']}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
            f.write(f"Num GPUs: {torch.cuda.device_count()}\n\n")
            
            f.write("="*80 + "\n")
            f.write("EXTRACTION STATISTICS\n")
            f.write("="*80 + "\n")
            f.write(f"Extraction Time: {extract_time:.2f}s\n")
            f.write(f"Num Calibration Samples: {len(calibration_set)}\n")
            f.write(f"Total Tokens Processed: ~{len(calibration_set) * 512:,}\n\n")
            
            f.write("="*80 + "\n")
            f.write("LAYER SENSITIVITIES\n")
            f.write("="*80 + "\n")
            f.write(f"{'Layer':<10} {'Sensitivity':<20} {'Rank':<10}\n")
            f.write("-"*80 + "\n")
            
            sorted_layers = sorted(sensitivities.items(), 
                                  key=lambda x: float(x[1]), reverse=True)
            for rank, (layer_name, sens) in enumerate(sorted_layers, 1):
                f.write(f"{layer_name:<10} {sens:<20.6f} {rank:<10}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("SENSITIVITY STATISTICS\n")
            f.write("="*80 + "\n")
            sens_values = np.array([v for v in sensitivities.values()])
            f.write(f"Mean Sensitivity: {sens_values.mean():.6f}\n")
            f.write(f"Min Sensitivity: {sens_values.min():.6f}\n")
            f.write(f"Max Sensitivity: {sens_values.max():.6f}\n")
            f.write(f"Std Deviation: {sens_values.std():.6f}\n\n")
            
            f.write("="*80 + "\n")
            f.write("INTERPRETATION\n")
            f.write("="*80 + "\n")
            f.write("Higher sensitivity = Layer is more critical to model output\n")
            f.write("Lower sensitivity = Layer can tolerate more aggressive quantization\n")
            f.write("\nUse these sensitivities in Phase 2 to assign different bit-widths\n")
            f.write("to layers for mixed-precision quantization.\n")
        
        print(f"✓ Saved {metric.upper()} sensitivities:")
        print(f"  JSON: {json_filename}")
        print(f"  TXT:  {txt_filename}")
    
    # ========== FINAL SUMMARY ==========
    print_section("PHASE 1 COMPLETE - SENSITIVITY ANALYSIS FINISHED")
    print(f"""
✓ Successfully completed sensitivity analysis for {selected_model}

Configuration Used:
  Model: {selected_model} ({num_layers} layers)
  Task: Language Modeling (PTQ)
  Dataset: WikiText-2
  Calibration: {calib_size} samples
  Metrics: {', '.join([m.upper() for m in metrics_to_use])}
  Extraction Time: {extract_time:.2f}s

Saved Files (in Sensitivities/ folder):
  - JSON files for programmatic use
  - TXT files with metadata and statistics

Next Step: Use these sensitivity files in Phase 2 to:
  - Cluster layers by sensitivity
  - Assign bit-widths to clusters
  - Apply mixed-precision quantization
  - Train/fine-tune on WikiText-2 training set
""")

if __name__ == "__main__":
    main()
