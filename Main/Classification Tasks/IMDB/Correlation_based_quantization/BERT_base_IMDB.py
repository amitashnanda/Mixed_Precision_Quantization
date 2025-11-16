"""
Comprehensive Mixed-Precision Quantization Pipeline for BERT on IMDB
=====================================================================

This script provides an interactive, end-to-end framework for:
1. Computing layer sensitivity (calibration + similarity metrics)
2. Clustering layers into precision groups
3. Quantizing and evaluating mixed-precision models

Features:
- Interactive user prompts at each stage
- Multiple similarity metrics (PWCCA, SVCCA, CKA)
- Multiple clustering strategies (K-means, Percentile, Hierarchical)
- Flexible calibration sizes (5k, 10k, 25k)
- Comprehensive evaluation and logging

Usage:
    python BERT_base_IMDB.py

Author: Mixed-Precision Quantization Team
Date: 2025
"""

import os
import re
import json
import time
import random
import argparse
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from sklearn.cross_decomposition import CCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
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
    """
    Display a prompt and get user input.
    
    Args:
        prompt_text: Question to ask
        options: List of valid options
        default: Default option if user presses Enter
    
    Returns:
        Selected option
    """
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
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

# ============================================================================
# SIMILARITY METRICS
# ============================================================================

def svcca(X, Y, topk=20, eps=1e-10):
    """
    Singular Vector Canonical Correlation Analysis.
    
    Args:
        X, Y: 2D numpy arrays (n_samples, dim)
        topk: Number of top singular vectors to keep
    
    Returns:
        SVCCA similarity in [0, 1]
    """
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
    """
    Projection-Weighted Canonical Correlation Analysis.
    
    Args:
        X, Y: 2D numpy arrays (n_samples, dim)
        energy_threshold: Energy threshold for SVD
        cca_max_components: Max CCA components
    
    Returns:
        PWCCA similarity in [-1, 1]
    """
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
    """
    Centered Kernel Alignment (CKA).
    
    Args:
        X, Y: 2D numpy arrays (n_samples, dim)
        debiased: Whether to use debiased version
    
    Returns:
        CKA similarity in [0, 1]
    """
    def centering(K):
        """Centering matrix."""
        n = K.shape[0]
        unit = np.ones([n, n]) / n
        I = np.eye(n)
        H = I - unit
        return np.dot(np.dot(H, K), H)

    def linear_hsic(X, Y):
        """Linear HSIC (Hilbert-Schmidt Independence Criterion)."""
        # Gram matrices
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
        # Debiased CKA
        cka_val = hsic_xy / np.sqrt(hsic_xx * hsic_yy)
    else:
        cka_val = hsic_xy / (np.sqrt(hsic_xx) * np.sqrt(hsic_yy) + 1e-10)

    return float(max(min(cka_val, 1.0), 0.0))

# ============================================================================
# DATA & FEATURE EXTRACTION
# ============================================================================

@torch.no_grad()
def extract_layer_outputs(dataloader, model, device, num_layers=None):
    """
    Extract hidden states from all layers.
    
    Returns:
        Dict mapping layer_0, layer_1, ... to numpy arrays (n_samples, hidden_dim)
    """
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
# CLUSTERING & BIT ALLOCATION
# ============================================================================

def kmeans_clustering(sensitivities, n_clusters=3):
    """
    Cluster layers using K-means.
    
    Returns:
        Dict mapping layer_idx -> bits
    """
    values = sensitivities.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(values)
    
    # Map clusters by mean sensitivity (descending)
    cluster_means = [(c, float(values[labels == c].mean())) 
                     for c in range(n_clusters)]
    cluster_means.sort(key=lambda x: x[1], reverse=True)
    
    return labels, cluster_means

def hierarchical_clustering(sensitivities, n_clusters=3):
    """Cluster layers using agglomerative (hierarchical) clustering."""
    values = sensitivities.reshape(-1, 1)
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = agg.fit_predict(values)
    
    cluster_means = [(c, float(values[labels == c].mean())) 
                     for c in range(n_clusters)]
    cluster_means.sort(key=lambda x: x[1], reverse=True)
    
    return labels, cluster_means

def percentile_clustering(sensitivities, n_clusters=3):
    """
    Cluster layers using percentile bucketing.
    Top X% -> high precision, Bottom X% -> low precision.
    """
    num_layers = len(sensitivities)
    layer_sens_pairs = [(i, sensitivities[i]) for i in range(num_layers)]
    layer_sens_pairs.sort(key=lambda x: x[1], reverse=True)
    
    cluster_size = num_layers // n_clusters
    labels = np.zeros(num_layers, dtype=int)
    cluster_means = []
    
    for cluster_id in range(n_clusters):
        start_idx = cluster_id * cluster_size
        end_idx = start_idx + cluster_size if cluster_id < n_clusters - 1 else num_layers
        
        cluster_layer_indices = [layer_sens_pairs[i][0] for i in range(start_idx, end_idx)]
        cluster_vals = [sensitivities[idx] for idx in cluster_layer_indices]
        
        for layer_idx in cluster_layer_indices:
            labels[layer_idx] = cluster_id
        
        cluster_means.append((cluster_id, float(np.mean(cluster_vals))))
    
    cluster_means.sort(key=lambda x: x[1], reverse=True)
    return labels, cluster_means

# ============================================================================
# QUANTIZATION
# ============================================================================

class LinearLSQ(nn.Module):
    """
    Learned Step Size Quantization (LSQ) for weight quantization.
    Per-tensor uniform quantization at inference time.
    """
    def __init__(self, linear: nn.Linear, nbits_w: int):
        super().__init__()
        self.inner = linear
        self.nbits_w = int(nbits_w)
        self.register_buffer("saved_weight", self.inner.weight.detach().clone())

    def _quantize_tensor(self, x, nbits):
        """Quantize tensor to nbits."""
        qmin = -(2 ** (nbits - 1))
        qmax = (2 ** (nbits - 1)) - 1
        min_val = x.min()
        max_val = x.max()
        scale = (max_val - min_val) / float(qmax - qmin)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, device=x.device, dtype=x.dtype)
        scale = torch.clamp(scale, min=1e-8)
        zero_point = qmin - min_val / scale
        q = torch.round(x / scale + zero_point)
        q = torch.clamp(q, qmin, qmax)
        deq = (q - zero_point) * scale
        return deq

    def calculate_weight_bits(self):
        """Calculate bit savings."""
        orig_bits = int(self.saved_weight.numel() * 32)
        quant_bits = int(self.saved_weight.numel() * self.nbits_w)
        return orig_bits, quant_bits

    def forward(self, x):
        qw = self._quantize_tensor(self.saved_weight, self.nbits_w).to(self.inner.weight.dtype)
        orig_weight_data = self.inner.weight.data.clone()
        self.inner.weight.data = qw
        try:
            out = self.inner(x)
        finally:
            self.inner.weight.data = orig_weight_data
        return out

def set_module_by_qualname(root: nn.Module, qualname: str, new_mod: nn.Module):
    """Set a module by its qualified name."""
    parts = qualname.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)

def quantize_linear_layers(model: nn.Module, layer_bits_map: dict):
    """
    Wrap all Linear layers with quantization according to layer_bits_map.
    
    Returns:
        (model, total_orig_bits, total_quant_bits)
    """
    total_orig, total_quant = 0, 0
    targets = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            m = re.search(r"\.layer\.(\d+)\.", name)
            layer_idx = int(m.group(1)) if m else -1
            nbits = layer_bits_map.get(layer_idx, 8)
            targets.append((name, module, nbits))
    
    for qualname, linear_mod, nbits in targets:
        wrapper = LinearLSQ(linear_mod, nbits)
        set_module_by_qualname(model, qualname, wrapper)
        o, q = wrapper.calculate_weight_bits()
        total_orig += o
        total_quant += q
    
    return model, total_orig, total_quant

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, dataloader, device):
    """Evaluate model on sentiment classification task."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    
    return accuracy, precision, recall, f1

def bits_to_mb(bits: int) -> float:
    """Convert bits to megabytes."""
    return bits / 8.0 / (1024.0 * 1024.0)

# ============================================================================
# INTERACTIVE PIPELINE
# ============================================================================

def run_sensitivity_analysis():
    """Interactive sensitivity analysis pipeline."""
    print_section("PHASE 1: SENSITIVITY ANALYSIS")
    print("This phase computes how sensitive each layer is to quantization.")
    
    set_seed(42)
    device = pick_device()
    
    # --- Load Model ---
    print("\n📦 Loading BERT model...")
    model_name = "textattack/bert-base-uncased-imdb"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    base_model = AutoModel.from_pretrained(model_name).to(device)
    
    # --- Calibration Size ---
    print_section("Step 1: Calibration Set Size")
    print("""
    Calibration activations are collected from training data to compute sensitivities.
    
    Options:
    - 5k: Fast, but noisier sensitivity estimates (good for quick prototyping)
    - 10k: Medium speed, moderate stability
    - 25k: RECOMMENDED - full IMDB train set, most stable estimates
    
    Recommendation: Start with 5k for fast iteration, then run full 25k for final results.
    """)
    
    calib_options = ["5k (fast, noisy)", "10k (medium)", "25k (full IMDB train, recommended)"]
    calib_choice = prompt_user("Select calibration size:", calib_options, default="25k (full IMDB train, recommended)")
    
    calib_size_map = {
        "5k (fast, noisy)": 5000,
        "10k (medium)": 10000,
        "25k (full IMDB train, recommended)": 25000
    }
    calib_size = calib_size_map[calib_choice]
    
    # --- Load & Prepare Calibration Data ---
    print(f"\n📥 Loading IMDB dataset (using TRAIN split for calibration)...")
    ds = load_dataset("imdb")
    calibration_set = ds["train"].select(range(min(calib_size, len(ds["train"]))))
    print(f"✓ Loaded {len(calibration_set)} examples for calibration")
    
    # --- Sampling Strategy ---
    print_section("Step 2: Sampling Strategy")
    print("""
    How to sample from calibration data:
    
    - Random: Simple, fast
    - Stratified: Balanced by label (50/50 pos/neg) - RECOMMENDED
    """)
    
    sampling_choice = prompt_user(
        "Select sampling strategy:",
        ["Random sampling", "Stratified sampling (recommended)"],
        default="Stratified sampling (recommended)"
    )
    
    if "Stratified" in sampling_choice:
        # Simple stratification: equal from pos and neg
        pos_examples = [ex for ex in calibration_set if ex["label"] == 1]
        neg_examples = [ex for ex in calibration_set if ex["label"] == 0]
        
        n_take = min(len(pos_examples), len(neg_examples), calib_size // 2)
        calibration_set = calibration_set.__class__(
            pos_examples[:n_take] + neg_examples[:n_take]
        )
        print(f"✓ Using stratified sampling: {len(calibration_set)} examples (50/50 label split)")
    
    # Tokenize
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
    
    # --- Similarity Metrics ---
    print_section("Step 3: Similarity Metric Selection")
    print("""
    How to measure representational similarity between layers:
    
    - PWCCA: Projection-Weighted CCA - good default, captures correlation
    - SVCCA: SVD + CCA variant - faster, controlled via topk
    - CKA: Centered Kernel Alignment - robust, widely used alternative
    
    Recommendation: PWCCA + CKA (use both for verification)
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
    
    # Extract layer outputs once (reusable for all metrics)
    print_section("Extracting Layer Outputs")
    print(f"Processing {len(calibration_set)} calibration examples...")
    t0 = time.time()
    layer_outputs = extract_layer_outputs(loader, base_model, device)
    extract_time = time.time() - t0
    print(f"✓ Extracted {len(layer_outputs)} layers in {extract_time:.2f}s")
    
    # Compute sensitivities for each metric
    num_layers = base_model.config.num_hidden_layers
    sensitivities_dict = {}
    
    for metric in metrics_to_use:
        print_section(f"Computing {metric.upper()}-based Sensitivities")
        sensitivities = {}
        
        if metric == "pwcca":
            for i in tqdm(range(1, num_layers + 1), desc=f"PWCCA for layer"):
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
            for i in tqdm(range(1, num_layers + 1), desc=f"SVCCA for layer"):
                key, sens = compute_layer_sensitivity_svcca(layer_outputs, i, topk=svcca_topk)
                enc_idx = i - 1
                sensitivities[f"layer_{enc_idx}"] = float(sens)
        
        elif metric == "cka":
            for i in tqdm(range(1, num_layers + 1), desc=f"CKA for layer"):
                key, sens = compute_layer_sensitivity_cka(layer_outputs, i)
                enc_idx = i - 1
                sensitivities[f"layer_{enc_idx}"] = float(sens)
        
        sensitivities_dict[metric] = sensitivities
        
        # Print sensitivities
        print(f"\n{metric.upper()} Sensitivities:")
        for layer_name in sorted(sensitivities, key=lambda x: int(x.split("_")[1])):
            print(f"  {layer_name}: {sensitivities[layer_name]:.6f}")
    
    # Save results
    os.makedirs("Sensitivities", exist_ok=True)
    
    for metric, sensitivities in sensitivities_dict.items():
        json_filename = f"layer_sensitivity_BERT_IMDB_{metric}.json"
        json_path = os.path.join("Sensitivities", json_filename)
        with open(json_path, "w") as f:
            json.dump(sensitivities, f, indent=2)
        
        txt_filename = f"layer_sensitivity_BERT_IMDB_{metric}.txt"
        txt_path = os.path.join("Sensitivities", txt_filename)
        with open(txt_path, "w") as f:
            f.write(f"# {metric.upper()} Sensitivities for BERT-base-uncased on IMDB\n")
            f.write(f"# Calibration: TRAIN set ({len(calibration_set)} examples, {calib_choice})\n")
            f.write(f"# Sampling: {sampling_choice}\n")
            f.write(f"model: {model_name}\n")
            f.write(f"metric: {metric}\n")
            f.write(f"calibration_samples: {len(calibration_set)}\n")
            f.write(f"extraction_time_s: {extract_time:.2f}\n")
            f.write(f"num_gpus: {torch.cuda.device_count()}\n")
            f.write(f"batch_size: {batch_size}\n")
            for k in sorted(sensitivities, key=lambda x: int(x.split("_")[1])):
                f.write(f"{k}\t{sensitivities[k]:.6f}\n")
        
        print(f"✓ Saved {metric.upper()} sensitivities to {json_path}")
    
    print_section("Phase 1 Complete")
    print(f"Saved sensitivities for metrics: {', '.join(metrics_to_use).upper()}")
    
    return sensitivities_dict

def run_quantization_evaluation():
    """Interactive quantization and evaluation pipeline."""
    print_section("PHASE 2: QUANTIZATION & EVALUATION")
    print("This phase clusters layers and applies mixed-precision quantization.")
    
    set_seed(42)
    device = pick_device()
    
    # Load fine-tuned model
    print("\n📦 Loading BERT model...")
    model_name = "textattack/bert-base-uncased-imdb"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # Load test set
    print("\n📥 Loading IMDB test set...")
    ds = load_dataset("imdb")
    
    # --- Evaluation Set Size ---
    print_section("Step 1: Evaluation Set Size")
    print("""
    Evaluation is performed on the TEST set to measure quantization impact.
    
    Options:
    - 5k: Fast iteration/debugging (use stratified samples for variance estimate)
    - 10k: Medium evaluation set
    - 25k: RECOMMENDED - full IMDB test set, for final reported numbers
    """)
    
    eval_options = ["5k (fast iteration)", "10k (medium)", "25k (full test, recommended)"]
    eval_choice = prompt_user(
        "Select evaluation set size:",
        eval_options,
        default="25k (full test, recommended)"
    )
    
    eval_size_map = {
        "5k (fast iteration)": 5000,
        "10k (medium)": 10000,
        "25k (full test, recommended)": 25000
    }
    eval_size = eval_size_map[eval_choice]
    test_set = ds["test"].select(range(min(eval_size, len(ds["test"]))))
    print(f"✓ Using {len(test_set)} examples from TEST set")
    
    # Tokenize test set
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    
    test_tokenized = test_set.map(tokenize_fn, batched=True)
    test_tokenized = test_tokenized.remove_columns(
        [c for c in test_tokenized.column_names if c not in ["input_ids", "attention_mask", "label"]]
    )
    test_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    batch_size = 32 if torch.cuda.device_count() > 1 else 16
    test_loader = DataLoader(
        test_tokenized, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # --- Load Sensitivities ---
    print_section("Step 2: Load Sensitivity File")
    print("Available sensitivity files:")
    sens_dir = Path("Sensitivities")
    sens_files = list(sens_dir.glob("layer_sensitivity_BERT_IMDB_*.json"))
    
    if not sens_files:
        print("❌ No sensitivity files found. Run Phase 1 first.")
        return
    
    sens_file_names = [f.name for f in sens_files]
    selected_file = prompt_user(
        "Select sensitivity file to use:",
        sens_file_names,
        default=sens_file_names[0]
    )
    
    sens_path = Path("Sensitivities") / selected_file
    with open(sens_path, "r") as f:
        sensitivities = json.load(f)
    
    # Build sensitivity vector
    num_layers = max([int(k.split("_")[1]) for k in sensitivities.keys()]) + 1
    sens_values = np.array([sensitivities[f"layer_{i}"] for i in range(num_layers)], dtype=np.float32)
    
    print(f"✓ Loaded sensitivities from {selected_file}")
    print(f"  Layers: {num_layers}")
    print(f"  Mean sensitivity: {sens_values.mean():.6f}")
    print(f"  Sensitivity range: [{sens_values.min():.6f}, {sens_values.max():.6f}]")
    
    # --- Clustering Strategy ---
    print_section("Step 3: Layer Clustering / Grouping Strategy")
    print("""
    How to group layers by sensitivity for bit allocation:
    
    - K-means: Automatic clustering into k groups based on sensitivity values
    - Percentile: Deterministic buckets (top X%, middle, bottom X%)
    - Hierarchical: Agglomerative clustering to reveal natural groupings
    
    Recommendation: Start with K-means (3-4 clusters), validate with others if needed.
    """)
    
    clustering_options = ["K-means (recommended)", "Percentile bucketing", "Hierarchical clustering"]
    clustering_choice = prompt_user(
        "Select clustering strategy:",
        clustering_options,
        default="K-means (recommended)"
    )
    
    # --- Number of Groups ---
    print_section("Step 4: Number of Precision Groups")
    print("""
    How many different precision levels (groups) to use:
    
    3 groups (good balance):
    - Non-finetuned BERT: use (16/8/4) bits
    - Finetuned BERT: use (32/16/8) bits
    
    4 groups (finer control):
    - Non-finetuned BERT: use (32/16/8/4) bits
    - Finetuned BERT: use (32/16/8/4) bits with top layers at 32
    
    Recommendation: Start with 3 groups, increase to 4 if accuracy drops are large.
    """)
    
    n_groups_options = ["3 groups (simpler)", "4 groups (finer control)"]
    n_groups_choice = prompt_user(
        "Select number of groups:",
        n_groups_options,
        default="3 groups (simpler)"
    )
    n_clusters = int(n_groups_choice.split()[0])
    
    # Perform clustering
    print(f"\nClustering {num_layers} layers into {n_clusters} groups...")
    
    if "K-means" in clustering_choice:
        labels, cluster_means = kmeans_clustering(sens_values, n_clusters=n_clusters)
        strategy_name = "kmeans"
    elif "Percentile" in clustering_choice:
        labels, cluster_means = percentile_clustering(sens_values, n_clusters=n_clusters)
        strategy_name = "percentile"
    else:  # Hierarchical
        labels, cluster_means = hierarchical_clustering(sens_values, n_clusters=n_clusters)
        strategy_name = "hierarchical"
    
    print(f"✓ Clustering complete. Cluster assignments:")
    for cluster_id, cluster_mean in cluster_means:
        layer_indices = [i for i in range(num_layers) if labels[i] == cluster_id]
        print(f"  Cluster {cluster_id} (sensitivity {cluster_mean:.6f}): layers {layer_indices}")
    
    # --- Bit Assignment ---
    print_section("Step 5: Bit Precision Assignment")
    print(f"""
    Assign bit-widths to each of the {n_clusters} clusters (from highest to lowest sensitivity):
    
    Common presets:
    
    For 3 groups:
    - Conservative: 16 / 8 / 4 bits (good accuracy, moderate compression)
    - Aggressive: 8 / 4 / 2 bits (smaller model, higher accuracy risk)
    
    For 4 groups:
    - Conservative: 32 / 16 / 8 / 4 bits (high accuracy, less compression)
    - Moderate: 16 / 8 / 4 / 2 bits (balanced)
    - Aggressive: 8 / 4 / 2 / 2 bits (small model)
    
    Recommendation: Start with conservative, iterate if needed.
    """)
    
    if n_clusters == 3:
        bit_options = [
            "[16, 8, 4] (conservative, recommended)",
            "[8, 4, 2] (aggressive)",
            "Custom (specify manually)"
        ]
        bit_choice = prompt_user(
            "Select bit allocation for clusters (high→low sensitivity):",
            bit_options,
            default="[16, 8, 4] (conservative, recommended)"
        )
        
        if "16, 8, 4" in bit_choice:
            cluster_bits = [16, 8, 4]
        elif "8, 4, 2" in bit_choice:
            cluster_bits = [8, 4, 2]
        else:
            bits_str = input("Enter bits as comma-separated (e.g., '16,8,4'): ").strip()
            cluster_bits = [int(b.strip()) for b in bits_str.split(",")]
    else:  # 4 groups
        bit_options = [
            "[32, 16, 8, 4] (conservative, recommended)",
            "[16, 8, 4, 2] (moderate)",
            "[8, 4, 2, 2] (aggressive)",
            "Custom (specify manually)"
        ]
        bit_choice = prompt_user(
            "Select bit allocation for clusters (high→low sensitivity):",
            bit_options,
            default="[32, 16, 8, 4] (conservative, recommended)"
        )
        
        if "32, 16, 8, 4" in bit_choice:
            cluster_bits = [32, 16, 8, 4]
        elif "16, 8, 4, 2" in bit_choice:
            cluster_bits = [16, 8, 4, 2]
        elif "8, 4, 2, 2" in bit_choice:
            cluster_bits = [8, 4, 2, 2]
        else:
            bits_str = input("Enter bits as comma-separated (e.g., '32,16,8,4'): ").strip()
            cluster_bits = [int(b.strip()) for b in bits_str.split(",")]
    
    # Build layer_bits_map
    layer_bits_map = {}
    for i in range(num_layers):
        cluster_id = labels[i]
        rank = next(j for j, (cid, _) in enumerate(cluster_means) if cid == cluster_id)
        layer_bits_map[i] = cluster_bits[rank]
    
    print(f"\n✓ Bit allocation:")
    for i in range(num_layers):
        print(f"  layer_{i}: {layer_bits_map[i]}-bit (cluster {labels[i]}, sensitivity {sensitivities[f'layer_{i}']:.6f})")
    
    # --- Evaluate Before Quantization ---
    print_section("Evaluating FP32 Baseline")
    print("Computing baseline metrics before quantization...")
    acc_before, prec_before, rec_before, f1_before = evaluate_model(model, test_loader, device)
    print(f"✓ Baseline FP32 Performance:")
    print(f"  Accuracy:  {acc_before:.4f}")
    print(f"  Precision: {prec_before:.4f}")
    print(f"  Recall:    {rec_before:.4f}")
    print(f"  F1-Score:  {f1_before:.4f}")
    
    # --- Apply Quantization ---
    print_section("Applying Mixed-Precision Quantization")
    print(f"Quantizing {num_layers} layers with assigned bit-widths...")
    
    tq0 = time.time()
    base_model = model.module if hasattr(model, 'module') else model
    model_q, orig_bits, quant_bits = quantize_linear_layers(base_model, layer_bits_map)
    
    if torch.cuda.device_count() > 1:
        model_q = torch.nn.DataParallel(model_q)
    
    tq1 = time.time()
    
    reduction_pct = 100.0 * (1.0 - (quant_bits / orig_bits)) if orig_bits else 0.0
    compression_ratio = (orig_bits / quant_bits) if quant_bits else float("inf")
    
    print(f"✓ Quantization complete in {tq1 - tq0:.3f}s")
    print(f"  Original size:  {bits_to_mb(orig_bits):.2f} MB")
    print(f"  Quantized size: {bits_to_mb(quant_bits):.2f} MB")
    print(f"  Compression:    {compression_ratio:.2f}x ({reduction_pct:.2f}% reduction)")
    
    # --- Evaluate After Quantization ---
    print_section("Evaluating Quantized Model")
    print("Computing metrics after quantization...")
    acc_after, prec_after, rec_after, f1_after = evaluate_model(model_q, test_loader, device)
    print(f"✓ Quantized Model Performance:")
    print(f"  Accuracy:  {acc_after:.4f}")
    print(f"  Precision: {prec_after:.4f}")
    print(f"  Recall:    {rec_after:.4f}")
    print(f"  F1-Score:  {f1_after:.4f}")
    
    # --- Performance Summary ---
    print_section("Quantization Impact Summary")
    acc_drop = acc_before - acc_after
    prec_drop = prec_before - prec_after
    rec_drop = rec_before - rec_after
    f1_drop = f1_before - f1_after
    
    print(f"Accuracy drop:  {acc_drop:.4f} ({acc_drop/acc_before*100:.2f}%)")
    print(f"Precision drop: {prec_drop:.4f}")
    print(f"Recall drop:    {rec_drop:.4f}")
    print(f"F1-Score drop:  {f1_drop:.4f}")
    
    if acc_drop > 0.05:
        print("\n⚠ Large accuracy drop detected (>5%). Consider:")
        print("  - Increasing bits for high-sensitivity layers")
        print("  - Using 4 groups instead of 3")
        print("  - Trying per-channel quantization")
        print("  - Fine-tuning with QAT (Quantization-Aware Training)")
    elif acc_drop > 0.02:
        print("\n⚠ Moderate accuracy drop (2-5%). Consider fine-tuning.")
    else:
        print("\n✓ Excellent: accuracy drop is minimal (<2%)")
    
    # Save results
    os.makedirs("Evaluation", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"eval_BERT_IMDB_{strategy_name}_{n_clusters}groups_{ts}.txt"
    log_path = os.path.join("Evaluation", log_filename)
    
    with open(log_path, "w") as f:
        f.write(f"# Mixed-Precision PTQ Results | IMDB | BERT-base-uncased\n")
        f.write(f"timestamp: {ts}\n")
        f.write(f"model_name: {model_name}\n")
        f.write(f"device: {device}\n")
        f.write(f"num_gpus: {torch.cuda.device_count()}\n\n")
        
        f.write(f"# Sensitivity Computation\n")
        f.write(f"sensitivity_file: {selected_file}\n")
        f.write(f"num_layers: {num_layers}\n\n")
        
        f.write(f"# Evaluation Set\n")
        f.write(f"eval_set_size: {len(test_set)}\n")
        f.write(f"batch_size: {batch_size}\n\n")
        
        f.write(f"# Clustering\n")
        f.write(f"clustering_method: {strategy_name}\n")
        f.write(f"num_clusters: {n_clusters}\n\n")
        
        f.write(f"# Bit Allocation\n")
        for i in range(num_layers):
            f.write(f"layer_{i}: {layer_bits_map[i]}-bit (sensitivity: {sensitivities[f'layer_{i}']:.6f})\n")
        
        f.write(f"\n# FP32 Baseline (Before Quantization)\n")
        f.write(f"accuracy: {acc_before:.6f}\n")
        f.write(f"precision: {prec_before:.6f}\n")
        f.write(f"recall: {rec_before:.6f}\n")
        f.write(f"f1_score: {f1_before:.6f}\n")
        
        f.write(f"\n# After Mixed-Precision Quantization\n")
        f.write(f"accuracy: {acc_after:.6f}\n")
        f.write(f"precision: {prec_after:.6f}\n")
        f.write(f"recall: {rec_after:.6f}\n")
        f.write(f"f1_score: {f1_after:.6f}\n")
        
        f.write(f"\n# Accuracy Drops\n")
        f.write(f"accuracy_drop: {acc_drop:.6f} ({acc_drop/acc_before*100:.2f}%)\n")
        f.write(f"precision_drop: {prec_drop:.6f}\n")
        f.write(f"recall_drop: {rec_drop:.6f}\n")
        f.write(f"f1_drop: {f1_drop:.6f}\n")
        
        f.write(f"\n# Compression Metrics\n")
        f.write(f"orig_bits: {orig_bits}\n")
        f.write(f"quant_bits: {quant_bits}\n")
        f.write(f"orig_size_mb: {bits_to_mb(orig_bits):.2f}\n")
        f.write(f"quant_size_mb: {bits_to_mb(quant_bits):.2f}\n")
        f.write(f"compression_ratio: {compression_ratio:.3f}x\n")
        f.write(f"size_reduction_pct: {reduction_pct:.2f}%\n")
        f.write(f"quantization_time_s: {tq1 - tq0:.3f}\n")
    
    print(f"\n✓ Results saved to: {log_path}")

def main():
    """Main interactive pipeline."""
    print_section("Mixed-Precision Quantization Framework for BERT on IMDB")
    print("""
    This framework enables end-to-end mixed-precision quantization with full control:
    
    1. PHASE 1: Sensitivity Analysis
       - Choose calibration size (5k, 10k, 25k)
       - Select similarity metrics (PWCCA, SVCCA, CKA)
       - Compute layer sensitivities
    
    2. PHASE 2: Quantization & Evaluation
       - Choose evaluation set size
       - Select clustering strategy (K-means, Percentile, Hierarchical)
       - Assign bit-widths to precision groups
       - Evaluate accuracy/compression tradeoffs
    
    Recommendation for quick start (5 min):
    - Phase 1: 5k calibration, PWCCA+CKA, default settings
    - Phase 2: 5k test, K-means 3 groups, [16/8/4] bits
    
    For final results (30 min):
    - Phase 1: 25k calibration, PWCCA+CKA, default settings
    - Phase 2: 25k test, K-means 4 groups, [32/16/8/4] bits
    """)
    
    phase_options = [
        "Phase 1: Sensitivity Analysis",
        "Phase 2: Quantization & Evaluation",
        "Run both phases (end-to-end)",
        "Exit"
    ]
    
    while True:
        phase_choice = prompt_user(
            "What would you like to do?",
            phase_options,
            default="Run both phases (end-to-end)"
        )
        
        if phase_choice == "Exit":
            print("\n✓ Goodbye!")
            break
        elif phase_choice == "Phase 1: Sensitivity Analysis":
            run_sensitivity_analysis()
        elif phase_choice == "Phase 2: Quantization & Evaluation":
            run_quantization_evaluation()
        elif phase_choice == "Run both phases (end-to-end)":
            run_sensitivity_analysis()
            proceed = input("\n✓ Phase 1 complete. Proceed to Phase 2? (y/n) [y]: ").strip().lower()
            if proceed != "n":
                run_quantization_evaluation()

if __name__ == "__main__":
    main()
