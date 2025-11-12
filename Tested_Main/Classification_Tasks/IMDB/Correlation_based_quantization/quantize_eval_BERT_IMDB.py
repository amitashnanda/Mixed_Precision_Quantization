# quantize_eval_BERT_IMDB.py
import os
import re
import time
import json
import random
import argparse
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import os, pathlib

HF_HOME = "/pscratch/sd/a/ananda/.cache/huggingface"
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_DATASETS_CACHE"] = f"{HF_HOME}/datasets"
os.environ["HF_HUB_CACHE"] = f"{HF_HOME}/hub"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

for p in (os.environ["HF_DATASETS_CACHE"], os.environ["HF_HUB_CACHE"]):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pick_device():
    if torch.cuda.is_available(): 
        print(f"CUDA available: {torch.cuda.device_count()} GPUs detected")
        return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

class LinearLSQ(nn.Module):
    """
    Per-tensor uniform quantization of weights to nbits at inference time.
    No training; no persistent change to underlying params.
    """
    def __init__(self, linear: nn.Linear, nbits_w: int):
        super().__init__()
        self.inner = linear
        self.nbits_w = int(nbits_w)
        self.register_buffer("saved_weight", self.inner.weight.detach().clone())

    def _quantize_tensor(self, x, nbits):
        qmin = -(2 ** (nbits - 1)); qmax = (2 ** (nbits - 1)) - 1
        min_val = x.min()
        max_val = x.max()
        scale = (max_val - min_val) / float(qmax - qmin)
        # Fix: Avoid creating tensor from tensor
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, device=x.device, dtype=x.dtype)
        scale = torch.clamp(scale, min=1e-8)
        zero_point = qmin - min_val / scale
        q = torch.round(x / scale + zero_point)
        q = torch.clamp(q, qmin, qmax)
        deq = (q - zero_point) * scale
        return deq

    def calculate_weight_bits(self):
        orig_bits = int(self.saved_weight.numel() * 32)
        quant_bits = int(self.saved_weight.numel() * self.nbits_w)
        return orig_bits, quant_bits

    def forward(self, x):
        # Quantize weights
        qw = self._quantize_tensor(self.saved_weight, self.nbits_w).to(self.inner.weight.dtype)
        
        # Create a copy of original weight data (not the Parameter object itself)
        orig_weight_data = self.inner.weight.data.clone()
        
        # Temporarily replace weight data with quantized version
        self.inner.weight.data = qw
        
        try:
            out = self.inner(x)
        finally:
            # Restore original weight data
            self.inner.weight.data = orig_weight_data
        
        return out

def set_module_by_qualname(root: nn.Module, qualname: str, new_mod: nn.Module):
    parts = qualname.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)

def quantize_linear_layers(model: nn.Module, layer_bits_map: dict):
    """
    Wrap nn.Linear modules with LinearLSQ according to BERT layer index.
    Returns model, total_orig_bits, total_quant_bits.
    """
    total_orig, total_quant = 0, 0
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Match BERT layers: bert.encoder.layer.X
            m = re.search(r"\.layer\.(\d+)\.", name)
            layer_idx = int(m.group(1)) if m else -1
            nbits = layer_bits_map.get(layer_idx, 8)
            targets.append((name, module, nbits))
    
    for qualname, linear_mod, nbits in targets:
        wrapper = LinearLSQ(linear_mod, nbits)
        set_module_by_qualname(model, qualname, wrapper)
        o, q = wrapper.calculate_weight_bits()
        total_orig += o; total_quant += q
    
    return model, total_orig, total_quant

def bits_to_mb(bits: int) -> float:
    return bits / 8.0 / (1024.0 * 1024.0)

def evaluate_model(model, dataloader, device):
    """
    Evaluate BERT model on sentiment classification.
    Returns accuracy, precision, recall, f1.
    """
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
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    return accuracy, precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description='Mixed-Precision Quantization for BERT on IMDB')
    parser.add_argument('--allocation', type=str, default='kmeans', choices=['kmeans', 'percentile'],
                        help='Bit allocation strategy: kmeans or percentile (default: kmeans)')
    parser.add_argument('--sensitivity_method', type=str, default='pwcca', choices=['pwcca', 'svcca'],
                        help='Sensitivity method used (affects which file to load): pwcca or svcca (default: pwcca)')
    args = parser.parse_args()

    set_seed(42)
    device = pick_device()
    print(f"device: {device}")
    print(f"Bit allocation strategy: {args.allocation}")
    print(f"Sensitivity method: {args.sensitivity_method.upper()}")

    # Use fine-tuned BERT model on IMDB
    model_name = "textattack/bert-base-uncased-imdb"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    
    # Wrap model with DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    # Load IMDB dataset - USE TEST SPLIT FOR EVALUATION
    ds = load_dataset("imdb")
    test_set = ds["test"]
    print(f"Evaluating on TEST set: {len(test_set)} examples")

    # Tokenize test set
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=512
        )

    test_tokenized = test_set.map(tokenize_fn, batched=True)
    test_tokenized = test_tokenized.remove_columns([c for c in test_tokenized.column_names if c not in ["input_ids", "attention_mask", "label"]])
    test_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    batch_size = 32 if torch.cuda.device_count() > 1 else 16
    print(f"Using batch size: {batch_size}")
    test_loader = DataLoader(test_tokenized, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Evaluate BEFORE quantization
    print("\n" + "="*50)
    print("Evaluating Fine-tuned BERT (FP32) Before Quantization")
    print("="*50)
    acc_before, prec_before, rec_before, f1_before = evaluate_model(model, test_loader, device)
    print(f"Accuracy:  {acc_before:.4f}")
    print(f"Precision: {prec_before:.4f}")
    print(f"Recall:    {rec_before:.4f}")
    print(f"F1-Score:  {f1_before:.4f}")

    # Load sensitivities (computed on TRAIN set) - with method-specific filename
    sens_filename = f"layer_sensitivity_BERT_IMDB_{args.sensitivity_method}.json"
    sens_path = os.path.join("Sensitivities", sens_filename)
    if not os.path.exists(sens_path):
        # Fallback to default filename
        sens_path = os.path.join("Sensitivities", "layer_sensitivity_BERT_IMDB.json")
        if not os.path.exists(sens_path):
            raise FileNotFoundError(
                f"Missing sensitivities file: {sens_filename}. "
                f"Run sensitivity_BERT_IMDB.py --method {args.sensitivity_method} on TRAIN set first."
            )
    
    with open(sens_path, "r") as f:
        sens = json.load(f)

    # Build vector of sensitivities
    num_layers = max([int(k.split("_")[1]) for k in sens.keys()]) + 1
    values = np.array([sens[f"layer_{i}"] for i in range(num_layers)], dtype=np.float32)
    
    # Bit allocation based on strategy
    if args.allocation == 'kmeans':
        print("\nUsing K-means clustering for bit allocation (3 clusters)")
        values_reshaped = values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(values_reshaped)
        labels = kmeans.labels_
        means = [(c, float(values_reshaped[labels==c].mean())) for c in range(3)]
        means.sort(key=lambda x: x[1], reverse=True)

        # Map clusters -> bits: Using only 16/8/4-bit (3 precision levels)
        cluster_to_bits = {
            means[0][0]: 16,  # Highest sensitivity
            means[1][0]: 8,   # Medium sensitivity
            means[2][0]: 4    # Lowest sensitivity
        }
        layer_bits_map = {i: int(cluster_to_bits[labels[i]]) for i in range(num_layers)}
        
    else:  # percentile
        print("\nUsing percentile-based allocation")
        # Sort layers by sensitivity (descending)
        layer_sens_pairs = [(i, sens[f"layer_{i}"]) for i in range(num_layers)]
        layer_sens_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Allocate bits based on percentiles
        # Top 25% -> 16-bit, Middle 50% -> 8-bit, Bottom 25% -> 4-bit
        top_25 = int(np.ceil(num_layers * 0.25))
        bottom_25 = int(np.floor(num_layers * 0.25))
        
        layer_bits_map = {}
        for idx, (layer_idx, layer_sens) in enumerate(layer_sens_pairs):
            if idx < top_25:
                layer_bits_map[layer_idx] = 32  # Top 25%
            elif idx >= num_layers - bottom_25:
                layer_bits_map[layer_idx] = 16   # Bottom 25%
            else:
                layer_bits_map[layer_idx] = 8   # Middle 50%

    print("\n" + "="*50)
    print("Applying Mixed-Precision Quantization")
    print("="*50)
    print(f"Allocation strategy: {args.allocation}")
    print("Bit allocation per layer:")
    for i in range(num_layers):
        print(f"  layer_{i}: {layer_bits_map[i]}-bit (sensitivity: {sens[f'layer_{i}']:.6f})")

    # Quantize (wrap linears) - handle DataParallel wrapper
    tq0 = time.time()
    base_model = model.module if hasattr(model, 'module') else model
    model_q, orig_bits, quant_bits = quantize_linear_layers(base_model, layer_bits_map)
    
    # Re-wrap with DataParallel if needed
    if torch.cuda.device_count() > 1:
        model_q = torch.nn.DataParallel(model_q)
    
    tq1 = time.time()

    reduction_pct = 100.0 * (1.0 - (quant_bits / orig_bits)) if orig_bits else 0.0
    compression_ratio = (orig_bits / quant_bits) if quant_bits else float("inf")

    print(f"\nQuantization time: {tq1 - tq0:.3f}s")
    print(f"Model size: {bits_to_mb(orig_bits):.2f} MB → {bits_to_mb(quant_bits):.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Size reduction: {reduction_pct:.2f}%")

    # Evaluate AFTER quantization
    print("\n" + "="*50)
    print("Evaluating Quantized BERT After Mixed-Precision PTQ")
    print("="*50)
    acc_after, prec_after, rec_after, f1_after = evaluate_model(model_q, test_loader, device)
    print(f"Accuracy:  {acc_after:.4f}")
    print(f"Precision: {prec_after:.4f}")
    print(f"Recall:    {rec_after:.4f}")
    print(f"F1-Score:  {f1_after:.4f}")

    # Compute drops
    acc_drop = acc_before - acc_after
    prec_drop = prec_before - prec_after
    rec_drop = rec_before - rec_after
    f1_drop = f1_before - f1_after

    print("\n" + "="*50)
    print("Performance Summary")
    print("="*50)
    print(f"Accuracy drop:  {acc_drop:.4f} ({acc_drop/acc_before*100:.2f}%)")
    print(f"Precision drop: {prec_drop:.4f}")
    print(f"Recall drop:    {rec_drop:.4f}")
    print(f"F1-Score drop:  {f1_drop:.4f}")

    # Latency / throughput benchmark
    model_q.eval()
    with torch.no_grad():
        # Warmup
        sample_batch = next(iter(test_loader))
        sample_input = sample_batch["input_ids"][:1].to(device)
        sample_mask = sample_batch["attention_mask"][:1].to(device)
        
        for _ in range(3):
            _ = model_q(input_ids=sample_input, attention_mask=sample_mask)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        steps = min(100, len(test_loader))
        t0 = time.time()
        for i, batch in enumerate(test_loader):
            if i >= steps: break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            _ = model_q(input_ids=input_ids, attention_mask=attention_mask)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        
        latency_per_batch = (t1 - t0) / steps
        samples_processed = steps * batch_size
        throughput = samples_processed / (t1 - t0) if (t1 - t0) > 0 else 0.0

    fp32_model_size_mb = bits_to_mb(orig_bits)
    quant_model_size_mb_est = bits_to_mb(quant_bits)

    # Log results
    os.makedirs("Evaluation", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"quantize_eval_BERT_IMDB_{args.allocation}_{args.sensitivity_method}.txt"
    log_path = os.path.join("Evaluation", log_filename)
    with open(log_path, "w") as f:
        f.write(f"# Mixed-Precision PTQ ({args.sensitivity_method.upper()}-based) | IMDB | BERT-base-uncased\n")
        f.write(f"# Calibration: TRAIN set (5000 examples)\n")
        f.write(f"# Evaluation: TEST set (25000 examples)\n")
        f.write(f"# Allocation strategy: {args.allocation}\n")
        f.write(f"# Sensitivity method: {args.sensitivity_method}\n")
        f.write(f"model_name: {model_name}\n")
        f.write(f"device: {device}\n")
        f.write(f"num_gpus: {torch.cuda.device_count()}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"test_samples: {len(test_set)}\n")
        f.write("\n# Metrics BEFORE Quantization (FP32):\n")
        f.write(f"acc_before: {acc_before:.6f}\n")
        f.write(f"precision_before: {prec_before:.6f}\n")
        f.write(f"recall_before: {rec_before:.6f}\n")
        f.write(f"f1_before: {f1_before:.6f}\n")
        f.write("\n# Metrics AFTER Quantization:\n")
        f.write(f"acc_after: {acc_after:.6f}\n")
        f.write(f"precision_after: {prec_after:.6f}\n")
        f.write(f"recall_after: {rec_after:.6f}\n")
        f.write(f"f1_after: {f1_after:.6f}\n")
        f.write("\n# Performance Drops:\n")
        f.write(f"acc_drop: {acc_drop:.6f} ({acc_drop/acc_before*100:.2f}%)\n")
        f.write(f"precision_drop: {prec_drop:.6f}\n")
        f.write(f"recall_drop: {rec_drop:.6f}\n")
        f.write(f"f1_drop: {f1_drop:.6f}\n")
        f.write("\n# Compression Metrics:\n")
        f.write(f"orig_bits: {orig_bits}\n")
        f.write(f"quant_bits: {quant_bits}\n")
        f.write(f"reduction_pct: {reduction_pct:.2f}\n")
        f.write(f"compression_ratio: {compression_ratio:.3f}\n")
        f.write(f"fp32_model_size_mb: {fp32_model_size_mb:.2f}\n")
        f.write(f"quant_model_size_mb: {quant_model_size_mb_est:.2f}\n")
        f.write(f"quantize_time_s: {tq1 - tq0:.3f}\n")
        f.write("\n# Performance Metrics:\n")
        f.write(f"latency_per_batch_s: {latency_per_batch:.5f}\n")
        f.write(f"throughput_samples_per_s: {throughput:.2f}\n")
        f.write("\n# Layer Bit Allocation:\n")
        for i in range(num_layers):
            f.write(f"  layer_{i}: {layer_bits_map[i]}-bit\n")
    
    print(f"\nResults saved to: {log_path}")
    print(f"GPUs used: {torch.cuda.device_count()}")

if __name__ == "__main__":
    main()
