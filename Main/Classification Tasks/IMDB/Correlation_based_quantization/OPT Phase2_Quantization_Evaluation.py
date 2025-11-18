"""
Phase 2: Mixed-Precision Quantization - Layer Clustering & Model Evaluation for OPT Models on IMDB
====================================================================================================

This script uses sensitivity files from Phase 1 to:
1. Load sensitivity values computed in Phase 1
2. Cluster layers based on sensitivity
3. Assign bit-widths to each cluster
4. Apply mixed-precision quantization
5. Evaluate accuracy before and after quantization

User has complete control over:
1. Evaluation set size (5k, 10k, 25k)
2. Clustering strategy (K-means, Percentile, Hierarchical)
3. Number of precision groups (3 or 4)
4. Bit-width allocation to clusters

Output: Evaluation files with detailed performance metrics

Usage:
    python Phase2_Quantization_Evaluation.py

Author: Mixed-Precision Quantization Team
Date: 2025
"""

import os
import re
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
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

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
# CLUSTERING STRATEGIES
# ============================================================================

def kmeans_clustering(sensitivities, n_clusters=3):
    """Cluster layers using K-means."""
    values = sensitivities.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(values)
    
    cluster_means = [(c, float(values[labels == c].mean())) 
                     for c in range(n_clusters)]
    cluster_means.sort(key=lambda x: x[1], reverse=True)
    
    return labels, cluster_means

def hierarchical_clustering(sensitivities, n_clusters=3):
    """Cluster layers using hierarchical (agglomerative) clustering."""
    values = sensitivities.reshape(-1, 1)
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = agg.fit_predict(values)
    
    cluster_means = [(c, float(values[labels == c].mean())) 
                     for c in range(n_clusters)]
    cluster_means.sort(key=lambda x: x[1], reverse=True)
    
    return labels, cluster_means

def percentile_clustering(sensitivities, n_clusters=3):
    """Cluster layers using percentile bucketing."""
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
    """Learned Step Size Quantization (LSQ) for weight quantization."""
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
    """Wrap all Linear layers with quantization."""
    total_orig, total_quant = 0, 0
    targets = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            m = re.search(r"\.layers\.(\d+)\.", name)
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
# MAIN PIPELINE
# ============================================================================

def main():
    """Main Phase 2 quantization and evaluation pipeline."""
    print_section("PHASE 2: QUANTIZATION & EVALUATION FOR OPT MODELS ON IMDB")
    print("""
    This phase uses sensitivity files from Phase 1 to:
    ✓ Load layer sensitivities
    ✓ Cluster layers by sensitivity
    ✓ Assign bit-widths to clusters
    ✓ Apply mixed-precision quantization
    ✓ Evaluate performance (accuracy, precision, recall, F1)
    
    Complete user control over:
    ✓ Sensitivity file selection
    ✓ Evaluation set size (5k, 10k, 25k)
    ✓ Clustering strategy (K-means, Percentile, Hierarchical)
    ✓ Number of precision groups (3 or 4)
    ✓ Bit-width allocation
    """)
    
    set_seed(42)
    device = pick_device()
    
    # ========== STEP 1: Load Sensitivity File ==========
    print_section("STEP 1: LOAD SENSITIVITY FILE")
    print("Looking for sensitivity files in 'Sensitivities/' folder...")
    
    sens_dir = Path("Sensitivities")
    if not sens_dir.exists():
        print("❌ 'Sensitivities' folder not found!")
        print("Please run Phase 1 first to generate sensitivity files.")
        return
    
    sens_files = list(sens_dir.glob("sens_*.json"))
    if not sens_files:
        print("❌ No sensitivity files found in Sensitivities/ folder!")
        print("Please run Phase 1 first.")
        return
    
    sens_file_names = [f.name for f in sens_files]
    print(f"\nFound {len(sens_file_names)} sensitivity file(s):\n")
    for i, fname in enumerate(sens_file_names, 1):
        print(f"  {i}. {fname}")
    
    selected_file = prompt_user(
        "Select sensitivity file to use:",
        sens_file_names,
        default=sens_file_names[0]
    )
    
    sens_path = Path("Sensitivities") / selected_file
    with open(sens_path, "r") as f:
        sensitivities = json.load(f)
    
    # Load metadata from TXT file
    txt_file = selected_file.replace(".json", ".txt")
    txt_path = Path("Sensitivities") / txt_file
    metadata = {}
    if txt_path.exists():
        with open(txt_path, "r") as f:
            for line in f:
                if line.startswith("Model: "):
                    metadata["model"] = line.split("Model: ")[1].strip()
                elif line.startswith("Calibration Set Size: "):
                    metadata["calib_size"] = line.split("Calibration Set Size: ")[1].strip()
                elif line.startswith("Sampling Strategy: "):
                    metadata["sampling"] = line.split("Sampling Strategy: ")[1].strip()
                elif line.startswith("Similarity Metric: "):
                    metadata["metric"] = line.split("Similarity Metric: ")[1].strip()
    
    print(f"\n✓ Loaded sensitivity file: {selected_file}")
    if metadata:
        print(f"  Model: {metadata.get('model', 'Unknown')}")
        print(f"  Calibration: {metadata.get('calib_size', 'Unknown')}")
        print(f"  Sampling: {metadata.get('sampling', 'Unknown')}")
        print(f"  Metric: {metadata.get('metric', 'Unknown')}")
    
    # Build sensitivity vector and get model info
    num_layers = max([int(k.split("_")[1]) for k in sensitivities.keys()]) + 1
    sens_values = np.array([sensitivities[f"layer_{i}"] for i in range(num_layers)], dtype=np.float32)
    
    print(f"\n✓ Sensitivity statistics:")
    print(f"  Layers: {num_layers}")
    print(f"  Mean: {sens_values.mean():.6f}")
    print(f"  Min:  {sens_values.min():.6f}")
    print(f"  Max:  {sens_values.max():.6f}")
    print(f"  Std:  {sens_values.std():.6f}")
    
    # Find OPT model based on num_layers
    selected_model = None
    model_config = None
    for model_name, config in OPT_MODELS.items():
        if config["num_layers"] == num_layers:
            selected_model = model_name
            model_config = config
            break
    
    if not selected_model:
        print(f"⚠ Could not match {num_layers} layers to any OPT model!")
        print("Please run Phase 1 again.")
        return
    
    print(f"✓ Identified model: {selected_model}")
    model_hf_name = model_config["model_name"]
    
    # ========== STEP 2: Load Model ==========
    print_section("STEP 2: LOADING MODEL")
    print(f"Loading {selected_model} from HuggingFace Hub...")
    tokenizer = AutoTokenizer.from_pretrained(model_hf_name, use_fast=True)
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_hf_name).to(device)
    except:
        print("⚠ Loading base model and adding custom classification head...")
        base = AutoModel.from_pretrained(model_hf_name).to(device)
        
        class CustomClassifier(nn.Module):
            def __init__(self, base_model, hidden_dim):
                super().__init__()
                self.base = base_model
                self.classifier = nn.Linear(hidden_dim, 2)
                self.logits = None
            
            def forward(self, input_ids, attention_mask):
                outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
                pooled = outputs.last_hidden_state[:, 0, :]
                self.logits = self.classifier(pooled)
                return type('obj', (object,), {'logits': self.logits})()
        
        model = CustomClassifier(base, model_config["hidden_dim"]).to(device)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    print(f"✓ Model loaded successfully")
    
    # ========== STEP 3: Evaluation Set Size ==========
    print_section("STEP 3: EVALUATION SET SIZE")
    print("""
    Evaluation is performed on TEST set to measure quantization impact.
    
    Options:
    - 5k: Fast iteration/debugging (2-3 min)
    - 10k: Medium evaluation (5-8 min)
    - 25k: Full IMDB test set, RECOMMENDED (10-15 min)
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
    
    # ========== STEP 4: Load Test Set ==========
    print_section("STEP 4: LOADING TEST SET")
    print(f"Loading IMDB test set...")
    ds = load_dataset("imdb")
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
    
    print(f"Tokenizing {len(test_set)} examples...")
    test_tokenized = test_set.map(tokenize_fn, batched=True)
    test_tokenized = test_tokenized.remove_columns(
        [c for c in test_tokenized.column_names if c not in ["input_ids", "attention_mask", "label"]]
    )
    test_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    batch_size = 64 if torch.cuda.device_count() > 1 else 16
    test_loader = DataLoader(
        test_tokenized, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    print(f"✓ Test set ready. Batch size: {batch_size}")
    
    # ========== STEP 5: Clustering Strategy ==========
    print_section("STEP 5: CLUSTERING STRATEGY")
    print("""
    How to group layers by sensitivity for bit allocation:
    
    K-means:
    - Automatic clustering into k groups
    - Based on sensitivity values
    - Pros: Fast, deterministic with fixed seed
    - Cons: Need to specify k
    - Recommended: 3-4 clusters
    
    Percentile Bucketing:
    - Deterministic splits based on sensitivity ranking
    - Top X% → high precision, Bottom X% → low precision
    - Pros: No randomness, balanced groups
    - Cons: May not match natural clusters
    
    Hierarchical (Agglomerative):
    - Bottom-up clustering revealing natural groupings
    - Pros: Reveals structure, no initialization randomness
    - Cons: Slower, less intuitive
    
    Recommendation: Start with K-means (faster, widely used)
    """)
    
    clustering_options = ["K-means (recommended)", "Percentile bucketing", "Hierarchical clustering"]
    clustering_choice = prompt_user(
        "Select clustering strategy:",
        clustering_options,
        default="K-means (recommended)"
    )
    
    # ========== STEP 6: Number of Groups ==========
    print_section("STEP 6: NUMBER OF PRECISION GROUPS")
    print("""
    How many different bit-widths to use:
    
    3 groups (simpler, faster):
    - High sensitivity → one bit-width
    - Medium sensitivity → one bit-width
    - Low sensitivity → one bit-width
    Example: [16, 8, 4] bits
    
    4 groups (finer control):
    - Highest sensitivity → highest bits
    - High sensitivity → high bits
    - Medium sensitivity → medium bits
    - Low sensitivity → low bits
    Example: [32, 16, 8, 4] bits
    
    Recommendation: Start with 3 groups, use 4 for finer control
    """)
    
    n_groups_options = ["3 groups (simpler)", "4 groups (finer control)"]
    n_groups_choice = prompt_user(
        "Select number of groups:",
        n_groups_options,
        default="3 groups (simpler)"
    )
    n_clusters = int(n_groups_choice.split()[0])
    
    # ========== STEP 7: Perform Clustering ==========
    print_section("STEP 7: CLUSTERING LAYERS")
    print(f"Clustering {num_layers} layers into {n_clusters} groups based on sensitivity...")
    
    if "K-means" in clustering_choice:
        labels, cluster_means = kmeans_clustering(sens_values, n_clusters=n_clusters)
        strategy_name = "kmeans"
    elif "Percentile" in clustering_choice:
        labels, cluster_means = percentile_clustering(sens_values, n_clusters=n_clusters)
        strategy_name = "percentile"
    else:
        labels, cluster_means = hierarchical_clustering(sens_values, n_clusters=n_clusters)
        strategy_name = "hierarchical"
    
    print(f"✓ Clustering complete. Cluster assignments:")
    for cluster_id, cluster_mean in cluster_means:
        layer_indices = [i for i in range(num_layers) if labels[i] == cluster_id]
        print(f"  Cluster {cluster_id} (sensitivity {cluster_mean:.6f}): {len(layer_indices)} layers")
        print(f"    Layers: {layer_indices}")
    
    # ========== STEP 8: Bit Allocation ==========
    print_section("STEP 8: BIT-WIDTH ALLOCATION")
    print(f"""
    Assign bit-widths to clusters (from highest to lowest sensitivity):
    
    For 3 groups:
    
    Conservative [16, 8, 4]:
    - High sensitivity: 16 bits (full precision)
    - Medium: 8 bits (standard quantization)
    - Low: 4 bits (aggressive)
    - Expected: ~2-3x compression, <1% accuracy drop
    
    Aggressive [8, 4, 2]:
    - High sensitivity: 8 bits
    - Medium: 4 bits
    - Low: 2 bits
    - Expected: ~4-6x compression, 1-3% accuracy drop
    
    For 4 groups:
    
    Conservative [32, 16, 8, 4]:
    - Highest: 32 bits (preserve exactly)
    - High: 16 bits
    - Medium: 8 bits
    - Low: 4 bits
    - Expected: ~2-4x compression, minimal drop
    
    Moderate [16, 8, 4, 2]:
    - Highest: 16 bits
    - High: 8 bits
    - Medium: 4 bits
    - Low: 2 bits
    - Expected: ~4-8x compression, 1-2% drop
    
    Aggressive [8, 4, 2, 2]:
    - Maximum compression
    - Expected: ~6-10x compression, 3-5% drop
    
    Recommendation: Start with conservative, iterate if needed
    """)
    
    if n_clusters == 3:
        bit_options = [
            "[16, 8, 4] (conservative, recommended)",
            "[8, 4, 2] (aggressive)",
            "Custom (specify manually)"
        ]
        bit_choice = prompt_user(
            "Select bit allocation (high→medium→low sensitivity):",
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
    else:
        bit_options = [
            "[32, 16, 8, 4] (conservative, recommended)",
            "[16, 8, 4, 2] (moderate)",
            "[8, 4, 2, 2] (aggressive)",
            "Custom (specify manually)"
        ]
        bit_choice = prompt_user(
            "Select bit allocation (highest→high→medium→low sensitivity):",
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
    
    print(f"\n✓ Bit allocation assigned:")
    print(f"{'Layer':<10} {'Bits':<10} {'Sensitivity':<15} {'Cluster':<10}")
    print("-" * 45)
    for i in range(num_layers):
        print(f"layer_{i:<3} {layer_bits_map[i]:<10} {sensitivities[f'layer_{i}']:<15.6f} {labels[i]:<10}")
    
    # ========== STEP 9: Evaluate FP32 Baseline ==========
    print_section("STEP 9: EVALUATING FP32 BASELINE")
    print("Computing baseline metrics (before quantization)...")
    
    acc_before, prec_before, rec_before, f1_before = evaluate_model(model, test_loader, device)
    
    print(f"\n✓ Baseline FP32 Performance:")
    print(f"  Accuracy:  {acc_before:.6f}")
    print(f"  Precision: {prec_before:.6f}")
    print(f"  Recall:    {rec_before:.6f}")
    print(f"  F1-Score:  {f1_before:.6f}")
    
    # ========== STEP 10: Apply Quantization ==========
    print_section("STEP 10: APPLYING QUANTIZATION")
    print(f"Quantizing {num_layers} layers with mixed-precision weights...")
    
    tq0 = time.time()
    base_model = model.module if hasattr(model, 'module') else model
    model_q, orig_bits, quant_bits = quantize_linear_layers(base_model, layer_bits_map)
    
    if torch.cuda.device_count() > 1:
        model_q = torch.nn.DataParallel(model_q)
    
    tq1 = time.time()
    quant_time = tq1 - tq0
    
    reduction_pct = 100.0 * (1.0 - (quant_bits / orig_bits)) if orig_bits else 0.0
    compression_ratio = (orig_bits / quant_bits) if quant_bits else float("inf")
    
    print(f"✓ Quantization complete in {quant_time:.3f}s")
    print(f"  Original size:  {bits_to_mb(orig_bits):.2f} MB")
    print(f"  Quantized size: {bits_to_mb(quant_bits):.2f} MB")
    print(f"  Compression:    {compression_ratio:.2f}x ({reduction_pct:.2f}% reduction)")
    
    # ========== STEP 11: Evaluate Quantized Model ==========
    print_section("STEP 11: EVALUATING QUANTIZED MODEL")
    print("Computing metrics after quantization...")
    
    acc_after, prec_after, rec_after, f1_after = evaluate_model(model_q, test_loader, device)
    
    print(f"\n✓ Quantized Model Performance:")
    print(f"  Accuracy:  {acc_after:.6f}")
    print(f"  Precision: {prec_after:.6f}")
    print(f"  Recall:    {rec_after:.6f}")
    print(f"  F1-Score:  {f1_after:.6f}")
    
    # ========== STEP 12: Performance Summary ==========
    print_section("STEP 12: QUANTIZATION IMPACT SUMMARY")
    
    acc_drop = acc_before - acc_after
    prec_drop = prec_before - prec_after
    rec_drop = rec_before - rec_after
    f1_drop = f1_before - f1_after
    
    print(f"\nPerformance Changes:")
    print(f"  Accuracy drop:  {acc_drop:.6f} ({acc_drop/acc_before*100:.2f}%)")
    print(f"  Precision drop: {prec_drop:.6f}")
    print(f"  Recall drop:    {rec_drop:.6f}")
    print(f"  F1-Score drop:  {f1_drop:.6f}")
    
    print(f"\nCompression Metrics:")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Size reduction: {reduction_pct:.2f}%")
    print(f"  Original size: {bits_to_mb(orig_bits):.2f} MB")
    print(f"  Quantized size: {bits_to_mb(quant_bits):.2f} MB")
    
    if acc_drop > 0.05:
        print("\n⚠ Large accuracy drop (>5%). Consider:")
        print("  - Increasing bits for high-sensitivity layers")
        print("  - Using 4 groups instead of 3")
        print("  - Fine-tuning with QAT")
    elif acc_drop > 0.02:
        print("\n⚠ Moderate accuracy drop (2-5%). Consider fine-tuning.")
    else:
        print("\n✓ Excellent: accuracy drop is minimal (<2%)")
    
    # ========== STEP 13: Save Results ==========
    print_section("STEP 13: SAVING EVALUATION RESULTS")
    
    os.makedirs("Evaluation", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_filename = f"eval_{selected_model}_{strategy_name}_{n_clusters}g_{timestamp}.txt"
    log_path = os.path.join("Evaluation", log_filename)
    
    with open(log_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MIXED-PRECISION QUANTIZATION EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("="*80 + "\n")
        f.write("MODEL & SENSITIVITY CONFIGURATION\n")
        f.write("="*80 + "\n")
        f.write(f"Model: {selected_model}\n")
        f.write(f"Model HF Hub: {model_hf_name}\n")
        f.write(f"Num Layers: {num_layers}\n")
        f.write(f"Sensitivity File: {selected_file}\n")
        if metadata:
            f.write(f"Sensitivity Calibration: {metadata.get('calib_size', 'Unknown')}\n")
            f.write(f"Sensitivity Sampling: {metadata.get('sampling', 'Unknown')}\n")
            f.write(f"Sensitivity Metric: {metadata.get('metric', 'Unknown')}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        
        f.write("="*80 + "\n")
        f.write("QUANTIZATION CONFIGURATION\n")
        f.write("="*80 + "\n")
        f.write(f"Evaluation Set Size: {eval_choice}\n")
        f.write(f"Evaluation Samples: {len(test_set)}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Clustering Method: {strategy_name}\n")
        f.write(f"Number of Groups: {n_clusters}\n")
        f.write(f"Bit Allocation: {cluster_bits}\n")
        f.write(f"Quantization Time: {quant_time:.3f}s\n\n")
        
        f.write("="*80 + "\n")
        f.write("CLUSTER ASSIGNMENTS\n")
        f.write("="*80 + "\n")
        for cluster_id, cluster_mean in cluster_means:
            layer_indices = [i for i in range(num_layers) if labels[i] == cluster_id]
            rank = next(j for j, (cid, _) in enumerate(cluster_means) if cid == cluster_id)
            bits = cluster_bits[rank]
            f.write(f"Cluster {cluster_id}: {bits}-bit | Sensitivity {cluster_mean:.6f}\n")
            f.write(f"  Layers ({len(layer_indices)}): {layer_indices}\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("PER-LAYER BIT ALLOCATION\n")
        f.write("="*80 + "\n")
        f.write(f"{'Layer':<12} {'Bits':<8} {'Sensitivity':<15} {'Cluster':<8}\n")
        f.write("-" * 80 + "\n")
        for i in range(num_layers):
            f.write(f"layer_{i:<5} {layer_bits_map[i]:<8} {sensitivities[f'layer_{i}']:<15.6f} {labels[i]:<8}\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("PERFORMANCE RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"\nFP32 Baseline (Before Quantization):\n")
        f.write(f"  Accuracy:  {acc_before:.6f}\n")
        f.write(f"  Precision: {prec_before:.6f}\n")
        f.write(f"  Recall:    {rec_before:.6f}\n")
        f.write(f"  F1-Score:  {f1_before:.6f}\n")
        
        f.write(f"\nQuantized Model (After Quantization):\n")
        f.write(f"  Accuracy:  {acc_after:.6f}\n")
        f.write(f"  Precision: {prec_after:.6f}\n")
        f.write(f"  Recall:    {rec_after:.6f}\n")
        f.write(f"  F1-Score:  {f1_after:.6f}\n")
        
        f.write(f"\nAccuracy Drops (Absolute Change):\n")
        f.write(f"  Accuracy:  {acc_drop:.6f} ({acc_drop/acc_before*100:+.2f}%)\n")
        f.write(f"  Precision: {prec_drop:.6f}\n")
        f.write(f"  Recall:    {rec_drop:.6f}\n")
        f.write(f"  F1-Score:  {f1_drop:.6f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("COMPRESSION METRICS\n")
        f.write("="*80 + "\n")
        f.write(f"Original Model Size (FP32):  {bits_to_mb(orig_bits):.2f} MB ({orig_bits} bits)\n")
        f.write(f"Quantized Model Size:        {bits_to_mb(quant_bits):.2f} MB ({quant_bits} bits)\n")
        f.write(f"Compression Ratio:           {compression_ratio:.2f}x\n")
        f.write(f"Size Reduction:              {reduction_pct:.2f}%\n")
        f.write(f"Quantization Time:           {quant_time:.3f}s\n")
        f.write(f"Device:                      {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"Num GPUs:                    {torch.cuda.device_count()}\n\n")
        
        f.write("="*80 + "\n")
        f.write("SENSITIVITY STATISTICS\n")
        f.write("="*80 + "\n")
        f.write(f"Mean Sensitivity:   {sens_values.mean():.6f}\n")
        f.write(f"Min Sensitivity:    {sens_values.min():.6f}\n")
        f.write(f"Max Sensitivity:    {sens_values.max():.6f}\n")
        f.write(f"Std Deviation:      {sens_values.std():.6f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*80 + "\n")
        if acc_drop > 0.05:
            f.write("• Large accuracy drop (>5%) detected\n")
            f.write("  → Increase bits for high-sensitivity layers\n")
            f.write("  → Use 4 groups instead of 3\n")
            f.write("  → Consider fine-tuning with QAT\n")
        elif acc_drop > 0.02:
            f.write("• Moderate accuracy drop (2-5%)\n")
            f.write("  → Fine-tuning recommended\n")
        else:
            f.write("• Excellent: accuracy drop is minimal (<2%)\n")
        
        if compression_ratio > 5:
            f.write(f"• High compression ({compression_ratio:.1f}x) achieved\n")
            if acc_drop < 0.01:
                f.write("  → Keep this configuration\n")
            else:
                f.write("  → Trade-off acceptable\n")
        elif compression_ratio > 3:
            f.write(f"• Good compression ({compression_ratio:.1f}x) achieved\n")
        else:
            f.write(f"• Modest compression ({compression_ratio:.1f}x)\n")
            f.write("  → Try more aggressive bit allocation\n")
    
    print(f"\n✓ Evaluation results saved to:")
    print(f"  {log_filename}")
    
    # ========== FINAL SUMMARY ==========
    print_section("PHASE 2 COMPLETE - QUANTIZATION EVALUATION FINISHED")
    print(f"""
✓ Successfully completed quantization and evaluation for {selected_model}

Results Summary:
  Model: {selected_model}
  Compression Ratio: {compression_ratio:.2f}x
  Size Reduction: {reduction_pct:.2f}%
  Accuracy Drop: {acc_drop/acc_before*100:.2f}%
  Quantization Time: {quant_time:.3f}s

Evaluation file saved: {log_filename}

To run another experiment:
1. Run this script again with different settings
2. Or run Phase 1 with a different model/metric
3. Compare evaluation files to find best configuration
""")

if __name__ == "__main__":
    main()
