# sensitivity_bert_large_HellaSwag.py
import os
import json
import time
import random
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from sklearn.cross_decomposition import CCA

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pick_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def pwcca(X, Y, energy_threshold=0.99, cca_max_components=20, eps=1e-10):
    """
    Lightweight PWCCA approximation:
    - center data, SVD-reduce each matrix keeping energy_threshold
    - apply CCA with <= cca_max_components components
    - weight canonical correlations by absolute x_weights
    """
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)

    def svd_reduce(A):
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        if S.size == 0:
            return A, S, Vt
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
        # fallback: correlation of means
        a = Xc.mean(axis=1); b = Yc.mean(axis=1)
        denom = np.std(a) * np.std(b)
        return float(np.corrcoef(a, b)[0,1]) if denom > eps else 0.0

    cca = CCA(n_components=n_comp, max_iter=1000)
    cca.fit(Xr, Yr)
    Xc_c, Yc_c = cca.transform(Xr, Yr)
    corrs = []
    for i in range(Xc_c.shape[1]):
        a = Xc_c[:, i]; b = Yc_c[:, i]
        denom = (np.std(a) * np.std(b))
        corr = float(np.corrcoef(a, b)[0,1]) if denom > eps else 0.0
        corrs.append(np.nan_to_num(corr))

    corrs = np.array(corrs)
    try:
        w = np.abs(cca.x_weights_).sum(axis=0)
        if np.all(w == 0): w = np.ones_like(w)
    except Exception:
        w = np.ones_like(corrs)
    w = w / (w.sum() + eps)
    return float((w * corrs).sum())

@torch.no_grad()
def extract_layer_outputs(dataloader, model, device):
    """
    Returns dict 'layer_0'..'layer_N' where layer_0 is embeddings (word embeddings projected),
    and layers 1..N are encoder layers. We use mean pooling across tokens with attention mask.
    """
    model.eval()
    num_layers_plus_embed = model.config.num_hidden_layers + 1
    layer_outputs = {f"layer_{i}": [] for i in range(num_layers_plus_embed)}
    for batch in tqdm(dataloader, desc="Collecting hidden states"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        hidden_states = out.hidden_states  # tuple length num_layers+1
        for i, hs in enumerate(hidden_states):
            # hs: (batch, seq_len, dim); attention_mask: (batch, seq_len)
            mask = attention_mask.unsqueeze(-1).to(hs.dtype)
            summed = (hs * mask).sum(dim=1)
            lens = mask.sum(dim=1).clamp(min=1.0)
            mean_pooled = (summed / lens).cpu().numpy()
            layer_outputs[f"layer_{i}"].append(mean_pooled)
    for k in layer_outputs:
        layer_outputs[k] = np.vstack(layer_outputs[k])
    return layer_outputs

def compute_layer_sensitivity_pwcca(layer_outputs, target_layer_index):
    target_key = f"layer_{target_layer_index}"
    target = layer_outputs[target_key]
    sims = []
    for key, data in layer_outputs.items():
        if key == target_key: continue
        n = min(len(target), len(data))
        sim = pwcca(target[:n], data[:n])
        sims.append(sim)
    mean_sim = float(np.mean(sims)) if sims else 0.0
    # sensitivity: higher when layer is less similar to others
    return target_key, 1.0 - mean_sim

def tokenize_example(tokenizer, ctx, endings, correct_index, max_length=128):
    # use correct ending for sensitivity per-sample (single input)
    chosen = endings[correct_index] if 0 <= correct_index < len(endings) else endings[0]
    text = ctx + " " + chosen
    return tokenizer(text, padding="max_length", truncation=True, max_length=max_length)

def main():
    set_seed(42)
    device = pick_device()
    print(f"device: {device}")

    model_name = "bert-large-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)

    ds = load_dataset("hellaswag")
    val = ds["validation"]

    # build tokenized dataset: use context + correct ending per example
    tokenized = {"input_ids": [], "attention_mask": []}
    for ex in tqdm(val, desc="Tokenizing val"):
        ctx = ex.get("ctx", "") if "ctx" in ex else ex.get("context","")
        endings = ex.get("endings", ex.get("ending", []))
        if isinstance(endings, str): endings = [endings]
        label = int(ex["label"])
        enc = tokenize_example(tokenizer, ctx, endings, label, max_length=128)
        tokenized["input_ids"].append(enc["input_ids"])
        tokenized["attention_mask"].append(enc["attention_mask"])

    # convert to tensors
    input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(input_ids, attention_mask)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    t0 = time.time()
    # adapt loader items to dict expected by extract_layer_outputs
    class WrapLoader:
        def __init__(self, dl):
            self.dl = dl
        def __iter__(self):
            for a,b in self.dl:
                yield {"input_ids": a, "attention_mask": b}
    layer_outputs = extract_layer_outputs(WrapLoader(loader), model, device)
    num_layers = model.config.num_hidden_layers  # e.g., 24 for bert-large

    layer_sens = {}
    for i in tqdm(range(1, num_layers + 1), desc="Computing PWCCA sensitivities"):
        _, sens = compute_layer_sensitivity_pwcca(layer_outputs, i)
        enc_idx = i - 1
        layer_sens[f"layer_{enc_idx}"] = float(sens)

    os.makedirs("Sensitivities", exist_ok=True)
    json_path = os.path.join("Sensitivities", "layer_senstivity_bert_large.json")
    with open(json_path, "w") as f:
        json.dump(layer_sens, f, indent=2)

    elapsed = time.time() - t0
    txt_path = os.path.join("Sensitivities", f"layer_senstivity_bert_large_runtime_{elapsed:.2f}s.txt")
    with open(txt_path, "w") as f:
        f.write("# PWCCA sensitivities for BERT-large (HellaSwag val)\n")
        f.write(f"time_s: {elapsed:.2f}\n")
        for k in sorted(layer_sens, key=lambda x: int(x.split("_")[1])):
            f.write(f"{k}\t{layer_sens[k]:.6f}\n")

    print(f"Saved JSON -> {json_path}")
    print(f"Saved TXT  -> {txt_path}")
    print(f"Time taken: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
