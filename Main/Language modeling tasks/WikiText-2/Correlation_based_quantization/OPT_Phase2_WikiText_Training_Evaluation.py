"""
Phase 2: Quantization, Training & Evaluation for OPT Models on WikiText-2 (Language Modeling)
==============================================================================================

This script uses sensitivity files from Phase 1 to:
1. Load sensitivity values computed in Phase 1
2. Cluster layers based on sensitivity
3. Assign bit-widths to each cluster
4. Apply mixed-precision quantization
5. Train quantized model on WikiText-2 training set
6. Evaluate on WikiText-2 validation set (language modeling - perplexity)

User has complete control over:
1. Sensitivity file selection
2. Clustering strategy (K-means, Percentile, Hierarchical)
3. Number of precision groups (3 or 4)
4. Bit-width allocation to clusters
5. Training parameters (learning rate, epochs, batch size)

Output: Training logs and evaluation metrics saved in Evaluation/ folder

Dataset: WikiText-2 (https://huggingface.co/datasets/Salesforce/wikitext)
Task: Language Modeling (Causal LM)
Metric: Perplexity (lower is better)

Usage:
    python OPT_Phase2_WikiText_Training_Evaluation.py

Author: Mixed-Precision Quantization Team
Date: 2025
"""

# ✅ ALL REQUIRED IMPORTS - MUST BE AT TOP
import os
import re
import json
import time
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AdamW
from sklearn.cluster import KMeans, AgglomerativeClustering

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
            value = int(user_input) if "." not in user_input else float(user_input)
            if min_val is not None and value < min_val:
                print(f"❌ Value must be >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"❌ Value must be <= {max_val}")
                continue
            return value
        except ValueError:
            print("❌ Invalid input.")

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
    """Cluster layers using hierarchical clustering."""
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
# CUSTOM DATASET FOR LANGUAGE MODELING
# ============================================================================

class WikiTextDataset(Dataset):
    """Custom dataset for WikiText language modeling."""
    def __init__(self, examples, max_length=512):
        self.examples = examples
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def calculate_perplexity(loss):
    """Calculate perplexity from loss."""
    return torch.exp(torch.tensor(loss))

def train_epoch(model, dataloader, optimizer, device, epoch, log_file=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - Training")
    
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Shift for language modeling: input and target
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({"loss": loss.item()})

        # ✅ ADD THIS BLOCK - Clear cache after each batch
        del outputs, loss, input_ids, attention_mask, labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    avg_loss = total_loss / num_batches
    perplexity = calculate_perplexity(avg_loss)
    
    log_msg = f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}"
    print(f"\n{log_msg}")
    if log_file:
        with open(log_file, "a") as f:
            f.write(log_msg + "\n")
    
    return avg_loss, perplexity.item()

@torch.no_grad()
def evaluate(model, dataloader, device, eval_type="Validation"):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"{eval_type}")
    
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Shift for language modeling
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({"loss": loss.item()})
    
        # ✅ ADD THIS BLOCK - Clear cache after each batch
        del outputs, loss, input_ids, attention_mask, labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    avg_loss = total_loss / num_batches
    perplexity = calculate_perplexity(avg_loss)
    
    return avg_loss, perplexity.item()

def bits_to_mb(bits: int) -> float:
    """Convert bits to megabytes."""
    return bits / 8.0 / (1024.0 * 1024.0)

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main Phase 2 quantization and training pipeline."""
    print_section("PHASE 2: QUANTIZATION & TRAINING FOR OPT MODELS ON WIKITEXT-2")
    print("""
    This phase uses sensitivity files from Phase 1 to:
    ✓ Load layer sensitivities
    ✓ Cluster layers by sensitivity
    ✓ Assign bit-widths to clusters
    ✓ Apply mixed-precision quantization
    ✓ Train quantized model on WikiText-2
    ✓ Evaluate with perplexity metric
    
    Complete user control over:
    ✓ Sensitivity file selection
    ✓ Clustering strategy (K-means, Percentile, Hierarchical)
    ✓ Number of precision groups (3 or 4)
    ✓ Bit-width allocation
    ✓ Training hyperparameters
    """)
    
    set_seed(42)
    device = pick_device()
    
    # ========== STEP 1: Load Sensitivity File ==========
    print_section("STEP 1: LOAD SENSITIVITY FILE")
    print("Looking for sensitivity files in 'Sensitivities/' folder...")
    
    sens_dir = Path("Sensitivities")
    if not sens_dir.exists():
        print("❌ 'Sensitivities' folder not found!")
        print("Please run Phase 1 first.")
        return
    
    sens_files = list(sens_dir.glob("sens_*.json"))
    if not sens_files:
        print("❌ No sensitivity files found!")
        return
    
    sens_file_names = [f.name for f in sens_files]
    print(f"Found {len(sens_file_names)} sensitivity file(s):\n")
    for i, fname in enumerate(sens_file_names, 1):
        print(f"  {i}. {fname}")
    
    selected_file = prompt_user(
        "Select sensitivity file:",
        sens_file_names,
        default=sens_file_names[0]
    )
    
    sens_path = Path("Sensitivities") / selected_file
    with open(sens_path, "r") as f:
        sensitivities = json.load(f)
    
    # Load metadata
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
    
    print(f"\n✓ Loaded: {selected_file}")
    if metadata:
        print(f"  Model: {metadata.get('model', 'Unknown')}")
        print(f"  Calibration: {metadata.get('calib_size', 'Unknown')}")
    
    # Build sensitivity vector
    num_layers = max([int(k.split("_")[1]) for k in sensitivities.keys()]) + 1
    sens_values = np.array([sensitivities[f"layer_{i}"] for i in range(num_layers)], dtype=np.float32)
    
    print(f"\n✓ Sensitivity statistics:")
    print(f"  Layers: {num_layers}")
    print(f"  Mean: {sens_values.mean():.6f}")
    print(f"  Min: {sens_values.min():.6f}")
    print(f"  Max: {sens_values.max():.6f}")
    
    # Match model
    selected_model = None
    model_config = None
    for model_name, config in OPT_MODELS.items():
        if config["num_layers"] == num_layers:
            selected_model = model_name
            model_config = config
            break
    
    if not selected_model:
        print(f"❌ Could not match {num_layers} layers to any OPT model!")
        return
    
    print(f"✓ Identified model: {selected_model}")
    model_hf_name = model_config["model_name"]
    
    # ========== STEP 2: Load Model ==========
    print_section("STEP 2: LOADING MODEL")
    print(f"Loading {selected_model} from HuggingFace Hub...")
    tokenizer = AutoTokenizer.from_pretrained(model_hf_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModel.from_pretrained(model_hf_name).to(device)
    
    # Add language modeling head
    class LMHead(nn.Module):
        def __init__(self, base_model, vocab_size, hidden_dim):
            super().__init__()
            self.model = base_model
            self.lm_head = nn.Linear(hidden_dim, vocab_size)
        
        def forward(self, input_ids, attention_mask, labels=None):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.last_hidden_state
            logits = self.lm_head(hidden_states)
            
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            
            return type('obj', (object,), {'loss': loss, 'logits': logits})()
    
    model = LMHead(model, tokenizer.vocab_size, model_config["hidden_dim"]).to(device)
    print(f"✓ Model loaded with LM head")
    
    # ========== STEP 3: Load WikiText-2 ==========
    print_section("STEP 3: LOADING WIKITEXT-2 DATASET")
    print("Loading WikiText-2...")
    
    try:
        ds = load_dataset("wikitext", "wikitext-2-v1")
    except:
        ds = load_dataset("wikitext", "wikitext-2")
    
    print(f"✓ Dataset loaded")
    print(f"  Train examples: {len(ds['train'])}")
    print(f"  Validation examples: {len(ds['validation'])}")
    print(f"  Test examples: {len(ds['test'])}")
    
    # ========== STEP 4: Prepare Data ==========
    print_section("STEP 4: PREPARING DATA")
    print("Tokenizing WikiText-2 for language modeling...")
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    
    train_tokenized = ds["train"].map(tokenize_fn, batched=True)
    train_tokenized = train_tokenized.remove_columns(
        [c for c in train_tokenized.column_names if c not in ["input_ids", "attention_mask"]]
    )
    train_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    val_tokenized = ds["validation"].map(tokenize_fn, batched=True)
    val_tokenized = val_tokenized.remove_columns(
        [c for c in val_tokenized.column_names if c not in ["input_ids", "attention_mask"]]
    )
    val_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # ========== STEP 5: Training Parameters ==========
    print_section("STEP 5: TRAINING PARAMETERS")
    print("""
    Configure training hyperparameters:
    
    Batch Size:
    - 8-16: If GPU memory is limited
    - 32: Standard for language modeling
    - 64: If you have ample GPU memory
    
    Learning Rate:
    - 1e-4: Standard for fine-tuning
    - 5e-5: More conservative
    
    Epochs:
    - 1: Quick training
    - 3: Standard (recommended)
    - 5: Longer training for better convergence
    """)
    
    batch_size = prompt_number(
        "Select batch size (8-64):",
        default=16,
        min_val=8,
        max_val=64
    )
    
    learning_rate = prompt_number(
        "Select learning rate (e.g., 0.0001 for 1e-4):",
        default=0.0001,
        min_val=1e-6,
        max_val=0.001
    )
    
    num_epochs = prompt_number(
        "Select number of training epochs (1-5):",
        default=3,
        min_val=1,
        max_val=5
    )
    
    print(f"\n✓ Training configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {num_epochs}")
    
    train_loader = DataLoader(train_tokenized, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tokenized, batch_size=batch_size, shuffle=False)
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # ========== STEP 6: Clustering Strategy ==========
    print_section("STEP 6: CLUSTERING STRATEGY")
    print("""
    K-means: Recommended, fast, deterministic
    Percentile: Deterministic bucketing
    Hierarchical: Reveals natural structure
    """)
    
    clustering_choice = prompt_user(
        "Select clustering strategy:",
        ["K-means (recommended)", "Percentile bucketing", "Hierarchical clustering"],
        default="K-means (recommended)"
    )
    
    # ========== STEP 7: Number of Groups ==========
    print_section("STEP 7: NUMBER OF PRECISION GROUPS")
    
    n_groups_choice = prompt_user(
        "Select number of groups:",
        ["3 groups (simpler)", "4 groups (finer control)"],
        default="3 groups (simpler)"
    )
    n_clusters = int(n_groups_choice.split()[0])
    
    # ========== STEP 8: Perform Clustering ==========
    print_section("STEP 8: CLUSTERING LAYERS")
    
    if "K-means" in clustering_choice:
        labels, cluster_means = kmeans_clustering(sens_values, n_clusters=n_clusters)
        strategy_name = "kmeans"
    elif "Percentile" in clustering_choice:
        labels, cluster_means = percentile_clustering(sens_values, n_clusters=n_clusters)
        strategy_name = "percentile"
    else:
        labels, cluster_means = hierarchical_clustering(sens_values, n_clusters=n_clusters)
        strategy_name = "hierarchical"
    
    print(f"✓ Clustering complete:")
    for cluster_id, cluster_mean in cluster_means:
        layer_indices = [i for i in range(num_layers) if labels[i] == cluster_id]
        print(f"  Cluster {cluster_id}: {len(layer_indices)} layers (sensitivity {cluster_mean:.6f})")
    
    # ========== STEP 9: Bit Allocation ==========
    print_section("STEP 9: BIT-WIDTH ALLOCATION")
    
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
    
    print(f"\n✓ Bit allocation assigned (sample of first 5 layers):")
    for i in range(min(5, num_layers)):
        print(f"  layer_{i}: {layer_bits_map[i]}-bit")
    print(f"  ... ({num_layers} total layers)")
    
    # ========== STEP 10: Apply Quantization ==========
    print_section("STEP 10: APPLYING QUANTIZATION")
    print("Quantizing model with mixed-precision weights...")
    
    tq0 = time.time()
    # Get the base model from LMHead wrapper
    base_model_for_quant = model.model
    base_model_for_quant, orig_bits, quant_bits = quantize_linear_layers(base_model_for_quant, layer_bits_map)
    model.model = base_model_for_quant
    tq1 = time.time()
    
    compression_ratio = (orig_bits / quant_bits) if quant_bits else float("inf")
    reduction_pct = 100.0 * (1.0 - (quant_bits / orig_bits)) if orig_bits else 0.0
    
    print(f"✓ Quantization complete in {tq1 - tq0:.3f}s")
    print(f"  Original size: {bits_to_mb(orig_bits):.2f} MB")
    print(f"  Quantized size: {bits_to_mb(quant_bits):.2f} MB")
    print(f"  Compression: {compression_ratio:.2f}x ({reduction_pct:.2f}% reduction)")
    
    # ========== STEP 11: Evaluate Before Training ==========
    print_section("STEP 11: EVALUATING BASELINE (BEFORE TRAINING)")
    print("Computing baseline perplexity on validation set...")
    
    model.eval()
    baseline_loss, baseline_ppl = evaluate(model, val_loader, device, "Baseline Validation")
    print(f"✓ Baseline - Loss: {baseline_loss:.4f}, Perplexity: {baseline_ppl:.4f}")
    
    # ✅ ADD THIS BLOCK - Clear cache before starting training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ Cleared GPU cache before training")
    
    # ========== STEP 12: Training Loop ==========
    print_section("STEP 12: TRAINING QUANTIZED MODEL")
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    os.makedirs("Evaluation", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"train_{selected_model}_WikiText2_{strategy_name}_{n_clusters}g_{timestamp}.txt"
    log_path = os.path.join("Evaluation", log_filename)
    
    # Initialize log file
    with open(log_path, "w") as f:
        f.write("="*80 + "\n")
        f.write("TRAINING LOG - QUANTIZED OPT MODEL ON WIKITEXT-2\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {selected_model}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Sensitivity File: {selected_file}\n")
        f.write(f"Clustering: {strategy_name} ({n_clusters} groups)\n")
        f.write(f"Bit Allocation: {cluster_bits}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write(f"Compression Ratio: {compression_ratio:.2f}x\n\n")
    
    print(f"Training for {num_epochs} epochs...")
    print(f"(Logging to {log_filename})\n")
    
    train_losses = []
    val_losses = []
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_ppl = train_epoch(model, train_loader, optimizer, device, epoch, log_path)
        train_losses.append(train_loss)
        
        val_loss, val_ppl = evaluate(model, val_loader, device, "Validation")
        val_losses.append(val_loss)
        
        with open(log_path, "a") as f:
            f.write(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Perplexity: {train_ppl:.4f}\n")
            f.write(f"Epoch {epoch} - Val Loss: {val_loss:.4f}, Perplexity: {val_ppl:.4f}\n\n")
    
        # ✅ ADD THIS BLOCK - Clear cache between epochs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"✓ Cleared GPU cache after Epoch {epoch}")
        
    # ========== STEP 13: Final Evaluation ==========
    print_section("STEP 13: FINAL EVALUATION")
    print("Computing final metrics after training...")
    
    model.eval()
    final_loss, final_ppl = evaluate(model, val_loader, device, "Final Validation")
    
    print(f"\n✓ Final Performance:")
    print(f"  Baseline Perplexity: {baseline_ppl:.4f}")
    print(f"  Final Perplexity: {final_ppl:.4f}")
    print(f"  Change: {final_ppl - baseline_ppl:+.4f}")
    
    # ========== STEP 14: Save Final Results ==========
    print_section("STEP 14: SAVING TRAINING RESULTS")
    
    with open(log_path, "a") as f:
        f.write("="*80 + "\n")
        f.write("FINAL RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Baseline Perplexity: {baseline_ppl:.4f}\n")
        f.write(f"Final Perplexity: {final_ppl:.4f}\n")
        f.write(f"Perplexity Change: {final_ppl - baseline_ppl:+.4f}\n")
        f.write(f"Model Size (Original): {bits_to_mb(orig_bits):.2f} MB\n")
        f.write(f"Model Size (Quantized): {bits_to_mb(quant_bits):.2f} MB\n")
        f.write(f"Compression Ratio: {compression_ratio:.2f}x\n")
        f.write(f"Size Reduction: {reduction_pct:.2f}%\n")
    
    print(f"\n✓ Training results saved to: {log_filename}")
    
    # ========== FINAL SUMMARY ==========
    print_section("PHASE 2 COMPLETE - TRAINING FINISHED")
    print(f"""
✓ Successfully completed quantization and training for {selected_model}

Training Summary:
  Model: {selected_model}
  Epochs: {num_epochs}
  Final Perplexity: {final_ppl:.4f}
  Baseline Perplexity: {baseline_ppl:.4f}
  Compression: {compression_ratio:.2f}x

Results file: {log_filename}
""")

if __name__ == "__main__":
    main()
