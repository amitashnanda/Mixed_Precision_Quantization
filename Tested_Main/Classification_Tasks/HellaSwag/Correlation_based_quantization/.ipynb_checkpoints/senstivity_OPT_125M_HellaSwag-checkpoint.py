# sensitivity_OPT_125M_HellaSwag.py
import os
import json
import time
import random
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

from sklearn.cross_decomposition import CCA

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

def prepare_example(tokenizer, ctx, ending, max_length=128):
    text = ctx + " " + ending
    return tokenizer(text, padding="max_length", truncation=True, max_length=max_length)

def pwcca(X, Y, energy_threshold=0.99, cca_max_components=20, eps=1e-10):
    """
    Approximate PWCCA similarity between X and Y.
    X, Y: 2D numpy arrays (n_samples, dim)
    Returns: pwcca similarity in [ -1, 1 ] (higher means more similar)
    """
    # center
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)

    # SVD reduce each to components keeping energy_threshold
    def svd_reduce(A):
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        if S.size == 0:
            return A, np.array([]), np.array([[]])
        energy = np.cumsum(S ** 2) / (np.sum(S ** 2) + eps)
        k = np.searchsorted(energy, energy_threshold) + 1
        k = min(k, S.size)
        return U[:, :k], S[:k], Vt[:k, :]

    # Project to principal subspaces (we'll work in original sample-projection)
    Ux, Sx, Vxt = svd_reduce(Xc)
    Uy, Sy, Vyt = svd_reduce(Yc)

    # If either reduced dim is 0, fallback to small cca
    Xr = Xc.dot(Vxt.T) if Vxt.size else Xc
    Yr = Yc.dot(Vyt.T) if Vyt.size else Yc

    n_comp = min(Xr.shape[1], Yr.shape[1], cca_max_components)
    if n_comp <= 0:
        # degenerate case: use simple corr between means
        try:
            return float(np.corrcoef(Xc.mean(axis=1), Yc.mean(axis=1))[0,1])
        except Exception:
            return 0.0

    cca = CCA(n_components=n_comp, max_iter=1000)
    cca.fit(Xr, Yr)
    Xc_c, Yc_c = cca.transform(Xr, Yr)

    # canonical correlations
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

    # importance weights from cca.x_weights_ (magnitude across features)
    try:
        w = np.abs(cca.x_weights_).sum(axis=0)
        if np.all(w == 0):
            w = np.ones_like(w)
    except Exception:
        w = np.ones_like(corrs)

    w = w / (w.sum() + eps)
    pwcca_val = float((w * corrs).sum())
    return pwcca_val

@torch.no_grad()
def extract_layer_outputs(dataloader, model, device):
    """
    Returns dict 'layer_0'..'layer_N' where layer_0 = embedding outputs,
    subsequent layers correspond to model.config.num_hidden_layers.
    Mean-pools across sequence length for each sample.
    """
    model.eval()
    
    # Wrap model with DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)
    
    num_layers_plus_embed = model.module.config.num_hidden_layers + 1 if hasattr(model, 'module') else model.config.num_hidden_layers + 1
    layer_outputs = {f"layer_{i}": [] for i in range(num_layers_plus_embed)}

    for batch in tqdm(dataloader, desc="Collecting hidden states"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        hidden_states = out.hidden_states  # tuple length num_layers+1
        # hidden_states[i]: (batch, seq_len, dim)
        for i, hs in enumerate(hidden_states):
            # mean pool over seq dim (respect attention mask)
            mask = attention_mask.unsqueeze(-1).to(hs.dtype)
            summed = (hs * mask).sum(dim=1)
            lens = mask.sum(dim=1).clamp(min=1.0)
            mean_pooled = (summed / lens).cpu().numpy()
            layer_outputs[f"layer_{i}"].append(mean_pooled)
    for k in layer_outputs:
        layer_outputs[k] = np.vstack(layer_outputs[k])
    return layer_outputs

def compute_layer_sensitivity_pwcca(layer_outputs, target_layer_index):
    """
    Sensitivity = 1 - mean PWCCA(target, other_layers).
    Higher sensitivity => layer is more unique/important.
    """
    target_key = f"layer_{target_layer_index}"
    target = layer_outputs[target_key]
    corrs = []
    for key, data in layer_outputs.items():
        if key == target_key: continue
        n = min(len(target), len(data))
        sim = pwcca(target[:n], data[:n])
        # clamp to [-1,1]
        sim = max(min(sim, 1.0), -1.0)
        corrs.append(sim)
    mean_corr = float(np.mean(corrs)) if corrs else 0.0
    return target_key, 1.0 - mean_corr

def main():
    set_seed(42)
    device = pick_device()
    print(f"device: {device}")

    model_name = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    base_model = AutoModel.from_pretrained(model_name).to(device)

    # Load HellaSwag validation set
    ds = load_dataset("hellaswag")
    val = ds["validation"]

    # For sensitivity: build input = ctx + correct ending (label indicates the correct choice)
    def tok_fn(ex):
        label = int(ex["label"])
        ctx = ex["context"] if "context" in ex else ex.get("ctx", "")
        endings = ex.get("endings", ex.get("ending", []))
        # ensure endings is a list
        if isinstance(endings, str):
            endings = [endings]
        correct_ending = endings[label] if 0 <= label < len(endings) else endings[0]
        return prepare_example(tokenizer, ctx, correct_ending, max_length=128)

    val = val.map(lambda ex: tok_fn(ex), batched=False)
    # keep only input_ids and attention_mask
    val = val.remove_columns([c for c in val.column_names if c not in ["input_ids", "attention_mask", "label"]])
    val.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Increase batch size to leverage multiple GPUs (4 GPUs * 16 = 64 effective batch size)
    batch_size = 64 if torch.cuda.device_count() > 1 else 16
    print(f"Using batch size: {batch_size}")
    loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    t0 = time.time()
    layer_outputs = extract_layer_outputs(loader, base_model, device)
    num_layers = base_model.config.num_hidden_layers  # N
    # compute sensitivity for layers 1..N -> map to layer_0..layer_{N-1} like your original pattern
    layer_sensitivities = {}
    for i in tqdm(range(1, num_layers + 1), desc="Computing PWCCA-based sensitivities"):
        key, sens = compute_layer_sensitivity_pwcca(layer_outputs, i)
        enc_idx = i - 1
        layer_sensitivities[f"layer_{enc_idx}"] = float(sens)

    os.makedirs("Sensitivities", exist_ok=True)
    json_path = os.path.join("Sensitivities", "layer_senstivity_OPT_125M.json")
    with open(json_path, "w") as f:
        json.dump(layer_sensitivities, f, indent=2)

    elapsed = time.time() - t0
    txt_path = os.path.join("Sensitivities", f"layer_senstivity_OPT_125M_runtime_{elapsed:.2f}s.txt")
    with open(txt_path, "w") as f:
        f.write(f"# PWCCA sensitivities for OPT-125M on HellaSwag (computed on val)\n")
        f.write(f"num_gpus: {torch.cuda.device_count()}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"time_s: {elapsed:.2f}\n")
        for k in sorted(layer_sensitivities, key=lambda x: int(x.split("_")[1])):
            f.write(f"{k}\t{layer_sensitivities[k]:.6f}\n")

    print(f"Saved sensitivities JSON -> {json_path}")
    print(f"Saved human-readable txt -> {txt_path}")
    print(f"Time taken: {elapsed:.2f}s")
    print(f"GPUs used: {torch.cuda.device_count()}")

if __name__ == "__main__":
    main()