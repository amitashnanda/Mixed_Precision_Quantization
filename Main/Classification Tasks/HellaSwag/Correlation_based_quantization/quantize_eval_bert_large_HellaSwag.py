# quantize_eval_bert_large_HellaSwag.py
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
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pick_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

class SimpleBertMultipleChoice(nn.Module):
    """
    Frozen BertModel encoder + small scoring head to score each choice.
    Head is linear: hidden_size -> 1. No fine-tuning here (inference-only).
    """
    def __init__(self, bert_model: BertModel):
        super().__init__()
        self.bert = bert_model
        self.cls = nn.Linear(self.bert.config.hidden_size, 1)
        # initialize deterministically
        torch.manual_seed(42)
        nn.init.normal_(self.cls.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.cls.bias)

    def forward(self, input_ids, attention_mask, labels=None):
        # input_ids: (batch, num_choices, seq_len)
        b, c, s = input_ids.shape
        flat_ids = input_ids.view(b * c, s)
        flat_att = attention_mask.view(b * c, s)
        out = self.bert(input_ids=flat_ids, attention_mask=flat_att, output_hidden_states=False, return_dict=True)
        # use pooled output [CLS] from BertModel (last_hidden_state[:,0,:])
        cls = out.last_hidden_state[:, 0, :]  # (b*c, dim)
        logits = self.cls(cls).view(b, c)    # (b, num_choices)
        loss = None
        if labels is not None:
            loss_f = nn.CrossEntropyLoss()
            loss = loss_f(logits, labels)
            return type("O", (), {"logits": logits, "loss": loss})
        return type("O", (), {"logits": logits, "loss": loss})

class LinearLSQ(nn.Module):
    """
    Per-tensor uniform quantization of weights to nbits at inference time.
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
        scale = max(scale, 1e-8)
        zero_point = qmin - float(min_val) / float(scale)
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
    total_orig, total_quant = 0, 0
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # try to infer layer index from BERT naming scheme
            m = re.search(r"encoder\.layer\.(\d+)\.", name)
            layer_idx = int(m.group(1)) if m else -1
            nbits = layer_bits_map.get(layer_idx, layer_bits_map.get(-1, 8))
            targets.append((name, module, nbits))
    for qualname, linear_mod, nbits in targets:
        wrapper = LinearLSQ(linear_mod, nbits)
        set_module_by_qualname(model, qualname, wrapper)
        o, q = wrapper.calculate_weight_bits()
        total_orig += o; total_quant += q
    return model, total_orig, total_quant

def bits_to_mb(bits: int) -> float:
    return bits / 8.0 / (1024.0 * 1024.0)

def prepare_tokenized_examples(val_ds, tokenizer, max_length=128):
    tokenized_examples = []
    for ex in tqdm(val_ds, desc="Tokenizing val"):
        ctx = ex.get("ctx", "") if "ctx" in ex else ex.get("context","")
        endings = ex.get("endings", ex.get("ending", []))
        if isinstance(endings, str): endings = [endings]
        texts = [ctx + " " + e for e in endings]
        enc = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        tokenized_examples.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "label": int(ex["label"])})
    return tokenized_examples

def gen_batches(tokenized_examples, batch_size=8):
    for i in range(0, len(tokenized_examples), batch_size):
        batch = tokenized_examples[i:i+batch_size]
        input_ids = torch.stack([it["input_ids"] for it in batch], dim=0)         # (batch, num_choices, seq_len)
        attention_mask = torch.stack([it["attention_mask"] for it in batch], dim=0)
        labels = torch.tensor([it["label"] for it in batch], dtype=torch.long)
        yield {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def evaluate_accuracy_from_generator(model, gen, device):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for batch in gen:
            inputs = { "input_ids": batch["input_ids"].to(device), "attention_mask": batch["attention_mask"].to(device) }
            labels = batch["labels"].to(device)
            out = model(**inputs)
            logits = out.logits
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0

def main():
    set_seed(42)
    device = pick_device()
    print(f"device: {device}")

    model_name = "bert-large-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert = BertModel.from_pretrained(model_name).to(device)
    model = SimpleBertMultipleChoice(bert).to(device)

    ds = load_dataset("hellaswag")
    val = ds["validation"]

    # prepare tokenized examples
    tokenized_examples = prepare_tokenized_examples(val, tokenizer, max_length=128)

    # evaluate before quantization
    acc_before = evaluate_accuracy_from_generator(model, gen_batches(tokenized_examples, batch_size=8), device)
    print(f"Accuracy before quantization: {acc_before:.4f}")

    # load sensitivities
    sens_path = os.path.join("Sensitivities", "layer_senstivity_bert_large.json")
    if not os.path.exists(sens_path):
        raise FileNotFoundError(f"Missing sensitivities file: {sens_path}. Run sensitivity script first.")
    with open(sens_path, "r") as f:
        sens = json.load(f)

    num_layers = bert.config.num_hidden_layers
    values = np.array([sens.get(f"layer_{i}", 0.0) for i in range(num_layers)], dtype=np.float32).reshape(-1,1)
    kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(values)
    labels = kmeans.labels_
    means = [(c, float(values[labels==c].mean())) for c in range(4)]
    means.sort(key=lambda x: x[1], reverse=True)
    cluster_to_bits = {means[0][0]:16, means[1][0]:8, means[2][0]:4, means[3][0]:2}
    layer_bits_map = {i: int(cluster_to_bits[labels[i]]) for i in range(num_layers)}
    # default bits for non-layer linears (-1) -> choose mid value 8
    layer_bits_map[-1] = 8

    # quantize
    tq0 = time.time()
    model_q, orig_bits, quant_bits = quantize_linear_layers(model, layer_bits_map)
    tq1 = time.time()

    reduction_pct = 100.0 * (1.0 - (quant_bits / orig_bits)) if orig_bits else 0.0
    compression_ratio = (orig_bits / quant_bits) if quant_bits else float("inf")

    # evaluate after quantization
    acc_after = evaluate_accuracy_from_generator(model_q, gen_batches(tokenized_examples, batch_size=8), device)
    acc_drop = acc_before - acc_after

    # latency & throughput approx (warm + fixed steps)
    model_q.eval()
    with torch.no_grad():
        first_batch = next(gen_batches(tokenized_examples, batch_size=8))
        for _ in range(3):
            _ = model_q(input_ids=first_batch["input_ids"].to(device), attention_mask=first_batch["attention_mask"].to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        steps = 32
        seen = 0
        t0 = time.time()
        for i, batch in enumerate(gen_batches(tokenized_examples, batch_size=8)):
            if i >= steps: break
            _ = model_q(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            seen += batch["input_ids"].size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        latency_per_batch = (t1 - t0) / max(1, min(steps, i+1))
        throughput = seen / (t1 - t0) if (t1 - t0) > 0 else 0.0

    fp32_model_size_mb = bits_to_mb(orig_bits)
    quant_model_size_mb_est = bits_to_mb(quant_bits)

    # write log
    os.makedirs("Evaluation", exist_ok=True)
    log_path = os.path.join("Evaluation", "quantize_eval_bert_large_HellaSwag.txt")
    with open(log_path, "w") as f:
        f.write("# Mixed-Precision PTQ (PWCCA-based) | HellaSwag | BERT-large\n")
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
