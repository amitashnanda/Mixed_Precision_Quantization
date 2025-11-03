# quantize_eval_OPT_125M_HellaSwag.py
import os
import re
import time
import json
import random
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMultipleChoice

from sklearn.cluster import KMeans

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pick_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def prepare_choices(tokenizer, ctx, endings, max_length=128):
    # returns a dict suitable to reshape into (num_choices, seq_len)
    texts = [ctx + " " + ending for ending in endings]
    enc = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    # enc.input_ids shape (num_choices, seq_len)
    return enc

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
        scale = torch.clamp(torch.tensor(scale, device=x.device, dtype=x.dtype), min=1e-8)
        zero_point = qmin - min_val / float(scale)
        q = torch.round(x / float(scale) + float(zero_point))
        q = torch.clamp(q, qmin, qmax)
        deq = (q - float(zero_point)) * float(scale)
        return deq

    def calculate_weight_bits(self):
        orig_bits = int(self.saved_weight.numel() * 32)
        quant_bits = int(self.saved_weight.numel() * self.nbits_w)
        return orig_bits, quant_bits

    def forward(self, x):
        qw = self._quantize_tensor(self.saved_weight, self.nbits_w).to(self.inner.weight.dtype)
        orig_w = self.inner.weight
        self.inner.weight = nn.Parameter(qw)
        try:
            out = self.inner(x)
        finally:
            self.inner.weight = orig_w
        return out

def set_module_by_qualname(root: nn.Module, qualname: str, new_mod: nn.Module):
    parts = qualname.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)

def quantize_linear_layers(model: nn.Module, layer_bits_map: dict):
    """
    Wrap nn.Linear modules with LinearLSQ according to encoder/decoder layer index.
    Returns model, total_orig_bits, total_quant_bits.
    """
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
        total_orig += o; total_quant += q
    return model, total_orig, total_quant

def bits_to_mb(bits: int) -> float:
    return bits / 8.0 / (1024.0 * 1024.0)

def prepare_mc_batch(tokenizer, batch_examples, device, max_length=128):
    """
    batch_examples: list of dataset examples (each contains context, endings list, label)
    returns dict suitable for model(**inputs) where:
    - input_ids: (batch_size, num_choices, seq_len)
    - attention_mask: same shape
    - labels: (batch_size,)
    """
    all_input_ids = []
    all_attention = []
    labels = []
    for ex in batch_examples:
        ctx = ex.get("context", ex.get("ctx", ""))
        endings = ex.get("endings", ex.get("ending", []))
        if isinstance(endings, str):
            endings = [endings]
        enc = prepare_choices(tokenizer, ctx, endings, max_length=max_length)
        all_input_ids.append(enc["input_ids"].unsqueeze(0))  # (1, num_choices, seq_len)
        all_attention.append(enc["attention_mask"].unsqueeze(0))
        labels.append(int(ex["label"]))
    input_ids = torch.cat(all_input_ids, dim=0).to(device)  # (batch, num_choices, seq_len)
    attention_mask = torch.cat(all_attention, dim=0).to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def evaluate_accuracy_mc(model, dataloader, device):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = {k: batch[k].to(device) if isinstance(batch[k], torch.Tensor) else batch[k] for k in ["input_ids","attention_mask"]}
            labels = batch["labels"].to(device)
            # model expects input_ids shape (batch, num_choices, seq_len)
            outputs = model(**inputs)
            logits = outputs.logits  # (batch, num_choices)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0

def main():
    set_seed(42)
    device = pick_device()
    print(f"device: {device}")

    model_name = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForMultipleChoice.from_pretrained(model_name).to(device)

    # HellaSwag validation
    ds = load_dataset("hellaswag")
    val = ds["validation"]

    # We'll prepare a DataLoader returning batches of pre-tokenized tensors
    # Build a list of tokenized examples (slower but simpler)
    tokenized_examples = []
    for ex in tqdm(val, desc="Tokenizing val"):
        ctx = ex.get("context", ex.get("ctx", ""))
        endings = ex.get("endings", ex.get("ending", []))
        if isinstance(endings, str):
            endings = [endings]
        enc = prepare_choices(tokenizer, ctx, endings, max_length=128)
        tokenized_examples.append({
            "input_ids": enc["input_ids"],           # (num_choices, seq_len) tensor
            "attention_mask": enc["attention_mask"], # (num_choices, seq_len)
            "label": int(ex["label"])
        })

    # Create a simple generator for DataLoader-like behavior
    batch_size = 8
    def gen_batches():
        for i in range(0, len(tokenized_examples), batch_size):
            batch_items = tokenized_examples[i:i+batch_size]
            # collate into tensors
            input_ids = torch.stack([it["input_ids"] for it in batch_items], dim=0)  # (batch, num_choices, seq_len)
            attention_mask = torch.stack([it["attention_mask"] for it in batch_items], dim=0)
            labels = torch.tensor([it["label"] for it in batch_items], dtype=torch.long)
            yield {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device), "labels": labels.to(device)}

    # Accuracy before
    def eval_from_generator(model, generator):
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for batch in tqdm(generator(), desc="Evaluating (gen)"):
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                logits = outputs.logits
                preds = logits.argmax(dim=-1)
                correct += (preds.cpu() == batch["labels"].cpu()).sum().item()
                total += batch["labels"].size(0)
        return correct / total if total else 0.0

    acc_before = eval_from_generator(model, gen_batches)
    print(f"Accuracy before quantization: {acc_before:.4f}")

    # Load sensitivities
    sens_path = os.path.join("Sensitivities", "layer_senstivity_OPT_125M.json")
    if not os.path.exists(sens_path):
        raise FileNotFoundError(f"Missing sensitivities file: {sens_path}. Run sensitivity script first.")
    with open(sens_path, "r") as f:
        sens = json.load(f)  # expects layer_0..layer_{N-1}

    # Build vector of sensitivities and cluster into 4 clusters
    num_layers = max([int(k.split("_")[1]) for k in sens.keys()]) + 1
    values = np.array([sens[f"layer_{i}"] for i in range(num_layers)], dtype=np.float32).reshape(-1, 1)
    kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(values)
    labels = kmeans.labels_
    means = [(c, float(values[labels==c].mean())) for c in range(4)]
    means.sort(key=lambda x: x[1], reverse=True)  # high sens first

    # Map clusters -> bits: most sensitive -> 16, then 8, then 4, then 2
    cluster_to_bits = {
        means[0][0]: 16,
        means[1][0]: 8,
        means[2][0]: 4,
        means[3][0]: 2
    }
    layer_bits_map = {i: int(cluster_to_bits[labels[i]]) for i in range(num_layers)}

    # Quantize (wrap linears)
    tq0 = time.time()
    model_q, orig_bits, quant_bits = quantize_linear_layers(model, layer_bits_map)
    tq1 = time.time()

    reduction_pct = 100.0 * (1.0 - (quant_bits / orig_bits)) if orig_bits else 0.0
    compression_ratio = (orig_bits / quant_bits) if quant_bits else float("inf")

    # Accuracy after
    # Re-evaluate using generator but with quantized model
    acc_after = eval_from_generator(model_q, gen_batches)
    acc_drop = acc_before - acc_after

    # Latency / throughput (approx): run a few steps
    model_q.eval()
    with torch.no_grad():
        # pick first batch from generator
        sample_gen = gen_batches()
        first_batch = next(sample_gen)
        # warmup
        for _ in range(3):
            _ = model_q(input_ids=first_batch["input_ids"], attention_mask=first_batch["attention_mask"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        steps = 32
        seen = 0
        t0 = time.time()
        for i, batch in enumerate(gen_batches()):
            if i >= steps: break
            _ = model_q(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            seen += batch["input_ids"].size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        latency_per_batch = (t1 - t0) / max(1, min(steps, i+1))
        throughput = seen / (t1 - t0) if (t1 - t0) > 0 else 0.0

    fp32_model_size_mb = bits_to_mb(orig_bits)
    quant_model_size_mb_est = bits_to_mb(quant_bits)

    # Log results
    os.makedirs("Evaluation", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("Evaluation", "quantize_eval_OPT_125M_HellaSwag.txt")
    with open(log_path, "w") as f:
        f.write("# Mixed-Precision PTQ (PWCCA-based) | HellaSwag | OPT-125M\n")
        f.write(f"model_name: {model_name}\n")
        f.write(f"device: {device}\n")
        f.write(f"acc_before: {acc_before:.6f}\n")
        f.write(f"acc_after: {acc_after:.6f}\n")
        f.write(f"acc_drop: {acc_drop:.6f}\n")
        f.write(f"orig_bits: {orig_bits}\n")
        f.write(f"quant_bits: {quant_bits}\n")
        f.write(f"reduction_pct: {reduction_pct:.2f}\n")
        f.write(f"compression_ratio: {compression_ratio:.3f}\n")
        f.write(f"fp32_model_size_mb (est): {fp32_model_size_mb:.2f}\n")
        f.write(f"quant_model_size_mb_est: {quant_model_size_mb_est:.2f}\n")
        f.write(f"quantize_time_s: {tq1 - tq0:.3f}\n")
        f.write(f"latency_per_batch_s(approx): {latency_per_batch:.5f}\n")
        f.write(f"throughput_samples_per_s(approx): {throughput:.2f}\n")
        f.write("layer_bits_map:\n")
        for i in range(num_layers):
            f.write(f"  layer_{i}: {layer_bits_map[i]}-bit\n")
    print(f"Saved log -> {log_path}")

if __name__ == "__main__":
    main()
