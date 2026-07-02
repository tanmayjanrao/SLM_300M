# ===========================================================================
# 300M LLM FROM SCRATCH
# Architecture : GPT-style Decoder-only Transformer
# Tokenizer    : tiktoken gpt2 (vocab=50257)
# Attention    : Flash Attention + RoPE + YaRN at inference
# Norm         : RMSNorm
# Activation   : SwiGLU
# Finetuning   : LoRA built-in (flip config flag)
# Hardware     : 1x T4 Kaggle (single GPU)
# ===========================================================================


# ===========================================================================
# CELL 1 — Install Dependencies
# ===========================================================================

# !pip install tiktoken datasets huggingface_hub -q

# tiktoken        → fast BPE tokenizer
# datasets        → HuggingFace datasets with Arrow streaming support
# huggingface_hub → HF login for authenticated dataset access


# ===========================================================================
# CELL 2 — Imports
# ===========================================================================

import os
import math
import time
import json
import gc
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from contextlib import nullcontext
import tiktoken
from datasets import load_dataset

# Single GPU setup
ddp        = False
local_rank = 0
world_size = 1
device     = "cuda" if torch.cuda.is_available() else "cpu"
master     = True

if master:
    print(f"Device     : {device}")
    print(f"DDP        : {ddp}  |  world_size={world_size}")
    if device != "cpu":
        print(f"GPU        : {torch.cuda.get_device_name(local_rank)}")
        print(f"VRAM       : {torch.cuda.get_device_properties(local_rank).total_memory/1e9:.1f} GB")
    print(f"PyTorch    : {torch.__version__}")


import os
# ===========================================================================
# HF AUTH (fixes unauthenticated download warnings + improves rate limits)
# ===========================================================================
os.environ["HF_TOKEN"] = "YourHFTOKEN"
from huggingface_hub import login
login("YourHFTOKEN")


# ===========================================================================
# CELL 3 — Configuration (single source of truth — only change here to scale)
# ===========================================================================

@dataclass
class ModelConfig:
    vocab_size   : int   = 50257
    context_len  : int   = 2048
    d_model      : int   = 1024
    n_heads      : int   = 16
    n_layers     : int   = 20
    # d_ff = 8/3 * d_model for SwiGLU (keeps param count same as 4x GELU MLP)
    # With d_model=1024 and n_layers=20, total model size is ~303M parameters.
    dropout      : float = 0.0      # 0 for large models — dropout hurts more than helps at scale
    bias         : bool  = False    # modern style — no bias in linear/norm layers

    # LoRA — False during pretraining, True during finetuning
    use_lora     : bool  = False
    lora_rank    : int   = 16
    lora_alpha   : float = 32.0

    @property
    def d_ff(self):
        # 8/3 * d_model rounded to nearest multiple of 256 for efficiency
        return ((int(8 * self.d_model / 3) + 255) // 256) * 256

    @property
    def head_dim(self):
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


@dataclass
class TrainConfig:
    # Dataset paths — update to your Kaggle input paths
    data_root : str = "/kaggle/input/llm-300m-tokenized"  # update this

    # Training
    batch_size      : int   = 16       # per GPU
    accum_steps     : int   = 8        # effective = batch * accum = 128
    max_steps       : int   = 120_000
    eval_interval   : int   = 500
    save_interval   : int   = 2_000
    sample_interval : int   = 2_000
    log_interval    : int   = 100

    # LR — cosine with warmup
    lr_peak         : float = 3e-4
    lr_min          : float = 3e-5
    warmup_steps    : int   = 1_000
    weight_decay    : float = 0.1
    grad_clip       : float = 1.0

    checkpoint_dir  : str   = "/kaggle/working/checkpoints_300M"
    resume_from     : Optional[str] = None


model_cfg = ModelConfig()
train_cfg = TrainConfig()

if master:
    # Rough param count
    emb   = model_cfg.vocab_size * model_cfg.d_model
    attn  = model_cfg.n_layers * (3 * model_cfg.d_model**2 + model_cfg.d_model**2)
    mlp   = model_cfg.n_layers * (3 * model_cfg.d_model * model_cfg.d_ff)  # SwiGLU has 3 matrices
    total = emb + attn + mlp
    print(f"\nMODEL CONFIG")
    print(f"  d_model={model_cfg.d_model}, n_heads={model_cfg.n_heads}, "
          f"n_layers={model_cfg.n_layers}, d_ff={model_cfg.d_ff}")
    print(f"  context_len={model_cfg.context_len}, vocab={model_cfg.vocab_size}")
    print(f"  ~{total/1e6:.0f}M parameters")
    print(f"\nTRAIN CONFIG")
    print(f"  effective batch = {train_cfg.batch_size} × {train_cfg.accum_steps} accum = "
          f"{train_cfg.batch_size * train_cfg.accum_steps}")
    print(f"  tokens per step = {train_cfg.batch_size * train_cfg.accum_steps * model_cfg.context_len:,}")


# ===========================================================================
# CELL 4 — Tokenizer
# ===========================================================================

enc = tiktoken.get_encoding("gpt2")

if master:
    sample = "Knowledge is the foundation of intelligence."
    tokens = enc.encode(sample)
    print(f"Sample  : '{sample}'")
    print(f"Tokens  : {tokens}")
    print(f"Decoded : {[enc.decode([t]) for t in tokens]}")
    print(f"Vocab   : {enc.n_vocab:,}  |  EOT token: {enc.eot_token}")


# ===========================================================================
# CELL 5 — Dataset (permanent Kaggle-mounted .bin files)
# ===========================================================================
# Files mounted at /kaggle/input/datasets/tjaycuz/slm-300m-tokens/
# Zero download. Zero combining. Instant on every session start.
# Val split taken proportionally from tail of each file.
# Memory safe — no eager batch loading, int64 cast only at getitem time.
# ===========================================================================

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR   = "/kaggle/input/datasets/tjaycuz/slm-300m-tokens"
DTYPE      = np.uint16
VAL_TOTAL  = 100_000_000   # 100M val tokens split proportionally across all files

BIN_FILES = [
    ("fineweb_tokens.bin",         2_800_000_000),
    ("wikipedia_tokens.bin",       1_600_000_000),
    ("fineweb_general_tokens.bin", 1_200_000_000),
    ("github_tokens.bin",          1_600_000_000),
    ("dclm_tokens.bin",              800_000_000),
]
EXPECTED_TOTAL = sum(n for _, n in BIN_FILES)  # 8_000_000_000

# Val slice per file — proportional to file size
# fineweb: 35M, wikipedia: 20M, fineweb_general: 15M, github: 20M, dclm: 10M
VAL_PER_FILE = {
    fname: int(VAL_TOTAL * (n / EXPECTED_TOTAL))
    for fname, n in BIN_FILES
}


# ── Step 1: Verify all files ──────────────────────────────────────────────────
if master:
    print("=" * 60)
    print("STEP 1 — VERIFYING FILES")
    print(f"  Dir   : {DATA_DIR}")
    print(f"  Total : {EXPECTED_TOTAL/1e9:.1f}B tokens expected")
    print("=" * 60)

full_paths = []
for fname, expected_tokens in BIN_FILES:
    path           = os.path.join(DATA_DIR, fname)
    expected_bytes = expected_tokens * 2

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n✗ Missing: {path}"
            f"\n  → Make sure tjaycuz/slm-300m-tokens is added via Kaggle Data panel"
        )

    actual_bytes  = os.path.getsize(path)
    actual_tokens = actual_bytes // 2
    val_slice     = VAL_PER_FILE[fname]
    status        = "✓" if actual_bytes == expected_bytes else "⚠ SIZE MISMATCH"

    if master:
        print(f"  {status}  {fname:<35}  "
              f"{actual_tokens/1e9:.3f}B total  "
              f"val={val_slice/1e6:.0f}M  "
              f"train={( actual_tokens - val_slice)/1e9:.3f}B")

    full_paths.append((path, actual_tokens))

total_tokens      = sum(n for _, n in full_paths)
total_val_tokens  = sum(VAL_PER_FILE.values())
total_train_tokens = total_tokens - total_val_tokens

if master:
    assert abs(total_tokens - EXPECTED_TOTAL) < 1_000_000, \
        f"Token count mismatch! got {total_tokens} expected {EXPECTED_TOTAL}"
    print(f"\n  Total tokens : {total_tokens/1e9:.3f}B  ✓")
    print(f"  Train tokens : {total_train_tokens/1e9:.3f}B  ({total_train_tokens/total_tokens*100:.1f}%)")
    print(f"  Val tokens   : {total_val_tokens/1e6:.0f}M  ({total_val_tokens/total_tokens*100:.1f}%)")

# ── Step 2: Dataset classes ───────────────────────────────────────────────────
class MultiBinDataset(Dataset):
    """
    Presents multiple .bin files as one continuous token stream.
    Each file contributes (total - val_slice) tokens for training.
    Val slice carved from tail of each file proportionally.
    Zero RAM. Zero disk copy. int64 cast only at read time.
    """
    def __init__(self, file_paths_and_lengths, context_len, val_per_file,
                 mode="train"):
        assert mode in ("train", "val")
        self.context_len = context_len
        self.mode        = mode
        self.mmaps       = []
        self.slices      = []   # (start, end) in each file's token space
        self.offsets     = []   # cumulative global offset per file
        cumulative       = 0

        for path, n_tokens in file_paths_and_lengths:
            fname     = os.path.basename(path)
            val_slice = val_per_file[fname]
            train_end = n_tokens - val_slice

            if mode == "train":
                start, end = 0, train_end
            else:
                start, end = train_end, n_tokens

            usable = end - start
            if usable <= 0:
                continue

            mm = np.memmap(path, dtype=DTYPE, mode="r", shape=(n_tokens,))
            self.mmaps.append(mm)
            self.slices.append((start, end))
            self.offsets.append(cumulative)
            cumulative += usable

        self.total_tokens = cumulative
        # Need context_len+1 tokens per window
        self.n_windows    = max(0, self.total_tokens - context_len)

    def __len__(self):
        return self.n_windows

    def _resolve(self, global_idx):
        """Map global window index → (file_idx, local_idx_within_slice)."""
        for i in range(len(self.mmaps) - 1, -1, -1):
            if global_idx >= self.offsets[i]:
                return i, global_idx - self.offsets[i]
        return 0, global_idx

    def _read_tokens(self, fi, local, count):
        """Read `count` tokens starting at local offset in file fi's slice."""
        start = self.slices[fi][0] + local
        end   = start + count
        slice_end = self.slices[fi][1]

        if end <= slice_end:
            # Fast path: entirely within one file
            return self.mmaps[fi][start:end].astype(np.int64)

        # Slow path: straddles file boundary
        tokens    = []
        remaining = count
        fj, lj   = fi, local
        while remaining > 0 and fj < len(self.mmaps):
            s, e      = self.slices[fj]
            avail     = (e - s) - lj
            take      = min(avail, remaining)
            abs_start = s + lj
            tokens.append(self.mmaps[fj][abs_start : abs_start + take].astype(np.int64))
            remaining -= take
            fj += 1
            lj  = 0
        return np.concatenate(tokens)

    def __getitem__(self, idx):
        fi, local = self._resolve(idx)
        chunk     = self._read_tokens(fi, local, self.context_len + 1)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


# ── Step 3: Build datasets ────────────────────────────────────────────────────
if master:
    print(f"\nSTEP 2 — BUILDING DATASETS")

train_dataset = MultiBinDataset(
    full_paths,
    context_len  = model_cfg.context_len,
    val_per_file = VAL_PER_FILE,
    mode         = "train",
)

val_dataset = MultiBinDataset(
    full_paths,
    context_len  = model_cfg.context_len,
    val_per_file = VAL_PER_FILE,
    mode         = "val",
)

if master:
    print(f"  Train windows : {len(train_dataset):,}")
    print(f"  Val windows   : {len(val_dataset):,}")
    print(f"  ✓ Datasets built — no data loaded into RAM yet")

# Free anything lingering before DataLoaders spin up workers
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ── Step 4: DataLoaders ───────────────────────────────────────────────────────
train_loader = DataLoader(
    train_dataset,
    batch_size         = train_cfg.batch_size,
    shuffle            = True,
    num_workers        = 1,        # single worker — single GPU, Kaggle ~13GB RAM
    pin_memory         = True,
    prefetch_factor    = 2,
    persistent_workers = True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size         = train_cfg.batch_size,
    shuffle            = False,
    num_workers        = 0,        # 0 for val — runs in main process, saves RAM
    pin_memory         = False,
)

if master:
    print(f"\nDATALOADER READY")
    print(f"  train batches/epoch : {len(train_loader):,}")
    print(f"  val   batches/epoch : {len(val_loader):,}")
    print(f"  batch shape         : ({train_cfg.batch_size}, {model_cfg.context_len})")
    print(f"\n✓ Cell 5 complete — cells 6-21 unchanged.")


# ===========================================================================
# CELL 6 — RMSNorm
# ===========================================================================
# Simpler than LayerNorm — normalizes by RMS only, no mean recentering.
# Faster, equally effective, used by LLaMA/Mistral/Gemma.
# Input/output shape: [B, T, d_model] → [B, T, d_model]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(d_model))  # learnable scale

    def forward(self, x):
        # x: [B, T, d_model]
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.scale * (x / rms)


# ===========================================================================
# CELL 7 — RoPE (Rotary Position Embeddings)
# ===========================================================================
# Encodes position by rotating Q and K vectors in attention.
# No position lookup table — position is baked into attention mathematically.
# Enables context extension at inference via YaRN scaling.
#
# How it works:
#   For each pair of dimensions (d0, d1) in head_dim:
#   rotate by angle = position × θ^(-2i/d)
#   Nearby tokens have similar rotations → high attention
#   Distant tokens have different rotations → naturally decay

def precompute_rope_freqs(head_dim, context_len, base=10000, device="cpu"):
    # θ frequencies: [head_dim/2]
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    # positions: [context_len]
    pos   = torch.arange(context_len, device=device).float()
    # outer product: [context_len, head_dim/2]
    freqs = torch.outer(pos, theta)
    # complex representation for efficient rotation
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # [T, head_dim/2]
    return freqs_cis


def apply_rope(q, k, freqs_cis):
    # q, k: [B, n_heads, T, head_dim]
    # freqs_cis: [T, head_dim/2]
    def rotate(x, freqs):
        # View as complex numbers — pairs of (real, imag)
        x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        # Broadcast freqs across batch and heads
        freqs = freqs.unsqueeze(0).unsqueeze(0)  # [1, 1, T, head_dim/2]
        x_rot = x_c * freqs
        return torch.view_as_real(x_rot).flatten(-2).to(q.dtype)

    T = q.shape[2]
    return rotate(q, freqs_cis[:T]), rotate(k, freqs_cis[:T])


# YaRN: scale RoPE frequencies at inference to extend context beyond training length
# Call this instead of precompute_rope_freqs when doing long-context inference
def precompute_rope_freqs_yarn(head_dim, context_len, base=10000,
                                scale=1.0, device="cpu"):
    # scale > 1.0 stretches the effective context window
    # scale=2.0 → effectively doubles context length
    # scale=4.0 → 4x context (train@2048 → infer@8192)
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    theta = theta / scale   # YaRN: divide frequencies by scale factor
    pos   = torch.arange(context_len, device=device).float()
    freqs = torch.outer(pos, theta)
    return torch.polar(torch.ones_like(freqs), freqs)


# ===========================================================================
# CELL 8 — LoRA Layer
# ===========================================================================
# Wraps nn.Linear with low-rank trainable adapters.
# Base weight frozen during finetuning — only A and B matrices train.
# Merge mode: bake A@B into base weight for zero-overhead inference.
#
# Output = base(x) + (lora_alpha/rank) * x @ A @ B
#   A: [d_model, rank]   initialized random
#   B: [rank, d_out]     initialized zeros (so adapter starts at identity)

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha, bias=False):
        super().__init__()
        self.base   = nn.Linear(in_features, out_features, bias=bias)
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scale  = alpha / rank
        self.rank   = rank

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A @ self.lora_B) * self.scale

    def merge_weights(self):
        # Merge LoRA into base weight for inference — no overhead at serving time
        with torch.no_grad():
            self.base.weight.data += (self.lora_A @ self.lora_B).T * self.scale
        del self.lora_A, self.lora_B


def make_linear(in_f, out_f, bias, cfg: ModelConfig):
    # Returns LoRALinear during finetuning, nn.Linear during pretraining
    if cfg.use_lora:
        return LoRALinear(in_f, out_f, cfg.lora_rank, cfg.lora_alpha, bias)
    return nn.Linear(in_f, out_f, bias=bias)


# ===========================================================================
# CELL 9 — SwiGLU MLP
# ===========================================================================
# Replaces standard GELU MLP. Three linear projections instead of two:
#   gate   = Linear(x)          → controls information flow
#   up     = Linear(x)          → projects up to d_ff
#   output = Linear(swish(gate) * up)  → project back to d_model
#
# d_ff = 8/3 * d_model to keep param count equivalent to 4x GELU MLP
# Shape: [B, T, d_model] → [B, T, d_model]

class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = make_linear(cfg.d_model, cfg.d_ff, cfg.bias, cfg)
        self.up   = make_linear(cfg.d_model, cfg.d_ff, cfg.bias, cfg)
        self.down = make_linear(cfg.d_ff, cfg.d_model, cfg.bias, cfg)

    def forward(self, x):
        # Swish(gate) * up — gated activation
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ===========================================================================
# CELL 10 — Causal Self-Attention with RoPE + Flash Attention
# ===========================================================================
# Shape trace:
#   Input          : [B, T, d_model]
#   QKV projection : [B, T, 3*d_model]  (single linear, split after)
#   Split heads    : [B, n_heads, T, head_dim]
#   Apply RoPE     : rotate Q and K by position — no shape change
#   Flash Attention: [B, n_heads, T, head_dim]  (is_causal=True, no manual mask)
#   Merge heads    : [B, T, d_model]
#   Out projection : [B, T, d_model]

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads  = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_model  = cfg.d_model

        # Single QKV projection — cleaner and faster than 3 separate linears
        self.qkv_proj = make_linear(cfg.d_model, 3 * cfg.d_model, cfg.bias, cfg)
        self.out_proj = make_linear(cfg.d_model, cfg.d_model,     cfg.bias, cfg)
        self.dropout  = cfg.dropout

    def forward(self, x, freqs_cis):
        B, T, C = x.shape

        # Project and split into Q, K, V
        qkv     = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # Reshape to multi-head format: [B, n_heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K — encodes position into attention
        q, k = apply_rope(q, k, freqs_cis)

        # Flash Attention — is_causal=True handles causal mask automatically
        # Dispatches to FlashAttention CUDA kernel on GPU
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # Merge heads: [B, n_heads, T, head_dim] → [B, T, d_model]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


# ===========================================================================
# CELL 11 — Transformer Block (Pre-LN + Attention + SwiGLU + Residuals)
# ===========================================================================
# Pre-LN: normalize BEFORE attention/MLP — much more stable than Post-LN
# Residuals: gradient highway through skip connections — enables deep networks
#
# x → RMSNorm → Attention → + x  (residual)
# x → RMSNorm → SwiGLU   → + x  (residual)

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1  = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = RMSNorm(cfg.d_model)
        self.mlp  = SwiGLU(cfg)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.ln1(x), freqs_cis)
        x = x + self.mlp(self.ln2(x))
        return x


# ===========================================================================
# CELL 12 — Full GPT Model
# ===========================================================================
# Forward pass:
#   token IDs [B, T]
#     → tok_emb [B, T, d_model]     (no pos_emb — RoPE handles position)
#     → TransformerBlock × 18       [B, T, d_model]
#     → RMSNorm                     [B, T, d_model]
#     → lm_head                     [B, T, vocab_size]
#     → CrossEntropyLoss            scalar

class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks   = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = RMSNorm(cfg.d_model)
        self.lm_head  = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying — lm_head shares tok_emb weights
        # Saves ~50M params and improves performance
        self.lm_head.weight = self.tok_emb.weight

        # Precompute RoPE frequencies — stored as buffer (not a parameter)
        freqs = precompute_rope_freqs(cfg.head_dim, cfg.context_len)
        self.register_buffer("freqs_cis", freqs)

        # GPT-2 style init — scale residual projections by 1/sqrt(2*n_layers)
        # Prevents variance explosion with deep residual networks
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down.base.weight") \
               or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear,)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def freeze_base_weights(self):
        # Called before finetuning — freeze everything except LoRA params
        for name, p in self.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Frozen base weights. Trainable (LoRA only): {trainable:,}")

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # ── Sequence-length assertion ─────────────────────────────────────────
        # Catches silent bugs where a batch longer than context_len is passed in.
        # RoPE freqs are only precomputed up to context_len — exceeding it gives
        # wrong positional encodings with no error, corrupting training silently.
        assert T <= self.cfg.context_len, (
            f"Input sequence length {T} exceeds context_len {self.cfg.context_len}. "
            f"Truncate inputs before calling forward()."
        )
        x    = self.tok_emb(idx)                       # [B, T, d_model]

        for block in self.blocks:
            x = block(x, self.freqs_cis)               # [B, T, d_model]

        x      = self.ln_final(x)                      # [B, T, d_model]
        logits = self.lm_head(x)                       # [B, T, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 yarn_scale=1.0):
        """
        Autoregressive generation with KV Cache + optional YaRN context extension.

        KV Cache: on the first forward pass we process the full prompt and cache
        every layer's K and V tensors. On each subsequent step we only run the
        single new token through the network, reading cached K/V for attention.
        This reduces generation from O(T²) to O(T) per step — critical for long
        outputs with a 300M model.

        yarn_scale > 1.0 extends context beyond training length at inference.
        """
        self.eval()

        # ── Choose freqs (standard or YaRN-extended) ─────────────────────────
        if yarn_scale != 1.0:
            extended_len   = int(self.cfg.context_len * yarn_scale)
            active_freqs   = precompute_rope_freqs_yarn(
                self.cfg.head_dim, extended_len,
                scale=yarn_scale, device=idx.device
            )
        else:
            extended_len = self.cfg.context_len
            active_freqs = self.freqs_cis

        B = idx.shape[0]

        # ── KV Cache: list of (k_cache, v_cache) per layer ───────────────────
        # Each cache starts empty; prefill step fills them with prompt K/V.
        # Shape after prefill: [B, n_heads, T_prompt, head_dim]
        kv_cache = [None] * len(self.blocks)   # one entry per TransformerBlock

        # ── Prefill: run the full prompt once, populate KV cache ─────────────
        x = self.tok_emb(idx)                                  # [B, T_prompt, d_model]
        T_prompt = idx.shape[1]

        for layer_idx, block in enumerate(self.blocks):
            # Run attention manually so we can intercept K and V
            residual = x
            x_norm   = block.ln1(x)

            # QKV projection
            B_, T_, C_ = x_norm.shape
            qkv     = block.attn.qkv_proj(x_norm)
            q, k, v = qkv.split(block.attn.d_model, dim=2)

            q = q.view(B_, T_, block.attn.n_heads, block.attn.head_dim).transpose(1, 2)
            k = k.view(B_, T_, block.attn.n_heads, block.attn.head_dim).transpose(1, 2)
            v = v.view(B_, T_, block.attn.n_heads, block.attn.head_dim).transpose(1, 2)

            q, k = apply_rope(q, k, active_freqs[:T_prompt])

            # Store K, V for this layer
            kv_cache[layer_idx] = (k, v)

            # Full causal attention over prompt
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2).contiguous().view(B_, T_, C_)
            x   = residual + block.attn.out_proj(out)
            x   = x + block.mlp(block.ln2(x))

        x      = self.ln_final(x)
        logits = self.lm_head(x)[:, -1, :]    # only last position needed

        # ── Decode loop: one token at a time, reusing cached K/V ─────────────
        generated = idx
        pos       = T_prompt   # current absolute position

        for _ in range(max_new_tokens):
            # Sample next token from logits
            logits = logits / temperature
            if top_k is not None:
                v_topk, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v_topk[:, [-1]]] = float("-inf")
            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)   # [B, 1]
            generated = torch.cat([generated, idx_next], dim=1)

            # Stop if we exceed max context
            if pos >= extended_len - 1:
                break

            # ── Single-token forward with KV cache ───────────────────────────
            x = self.tok_emb(idx_next)                           # [B, 1, d_model]

            for layer_idx, block in enumerate(self.blocks):
                residual = x
                x_norm   = block.ln1(x)

                B_, T_, C_ = x_norm.shape   # T_ == 1
                qkv     = block.attn.qkv_proj(x_norm)
                q, k, v = qkv.split(block.attn.d_model, dim=2)

                q = q.view(B_, 1, block.attn.n_heads, block.attn.head_dim).transpose(1, 2)
                k = k.view(B_, 1, block.attn.n_heads, block.attn.head_dim).transpose(1, 2)
                v = v.view(B_, 1, block.attn.n_heads, block.attn.head_dim).transpose(1, 2)

                # Apply RoPE at the current absolute position only
                q, k = apply_rope(q, k, active_freqs[pos : pos + 1])

                # Append new K, V to cache
                k_cache, v_cache = kv_cache[layer_idx]
                k_cache = torch.cat([k_cache, k], dim=2)   # [B, n_heads, pos+1, head_dim]
                v_cache = torch.cat([v_cache, v], dim=2)
                kv_cache[layer_idx] = (k_cache, v_cache)

                # Attend over full history (no causal mask needed — query is 1 token)
                out = F.scaled_dot_product_attention(q, k_cache, v_cache, is_causal=False)
                out = out.transpose(1, 2).contiguous().view(B_, 1, C_)
                x   = residual + block.attn.out_proj(out)
                x   = x + block.mlp(block.ln2(x))

            x      = self.ln_final(x)
            logits = self.lm_head(x)[:, -1, :]   # [B, vocab_size]
            pos   += 1

        return generated


# Build model
model = GPT(model_cfg).to(device)

# torch.compile — disabled on Kaggle T4: compilation spikes CPU RAM ~8GB and
# crashes the 13GB RAM limit before training even starts. T4 is also too old
# (Volta/Turing arch) to benefit much from compile anyway — speedup is minimal.
# Re-enable only on A100/H100 where RAM headroom and architecture support it.
# model = torch.compile(model)

raw_model = model   # single GPU — no DDP wrapper needed

if master:
    total     = sum(p.numel() for p in raw_model.parameters())
    trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    print(f"\nMODEL BUILT")
    print(f"  Total params     : {total/1e6:.1f}M")
    print(f"  Trainable params : {trainable/1e6:.1f}M")

    # Shape trace
    _x = torch.randint(0, model_cfg.vocab_size, (2, 64)).to(device)
    _y = torch.randint(0, model_cfg.vocab_size, (2, 64)).to(device)
    with torch.amp.autocast("cuda", enabled=(device != "cpu")):
        _logits, _loss = raw_model(_x, _y)
    print(f"  Input  shape     : {tuple(_x.shape)}")
    print(f"  Logits shape     : {tuple(_logits.shape)}  ← [B, T, vocab={model_cfg.vocab_size}]")
    print(f"  Loss at init     : {_loss.item():.3f}  (expect ~{math.log(model_cfg.vocab_size):.2f})")
    del _x, _y, _logits, _loss


# ===========================================================================
# CELL 13 — LR Schedule (cosine with warmup)
# ===========================================================================

def get_lr(step, cfg: TrainConfig):
    # Linear warmup: steps 0 → warmup_steps
    if step < cfg.warmup_steps:
        return cfg.lr_peak * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.lr_min
    # Cosine decay: warmup_steps → max_steps
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr_min + cosine * (cfg.lr_peak - cfg.lr_min)

if master:
    print("LR SCHEDULE")
    for s in [0, train_cfg.warmup_steps, train_cfg.max_steps//2, train_cfg.max_steps]:
        print(f"  step {s:>7} → lr = {get_lr(s, train_cfg):.2e}")


# ===========================================================================
# CELL 14 — Optimizer
# ===========================================================================
# fused=True: single GPU kernel for weight update — faster
# Weight decay on matrices only — NOT on biases, RMSNorm scale, embeddings

def configure_optimizer(model, cfg: TrainConfig):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)

    groups = [
        {"params": decay,    "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    try:
        opt = torch.optim.AdamW(groups, lr=cfg.lr_peak, betas=(0.9, 0.95),
                                 eps=1e-8, fused=True)
        if master: print("  AdamW fused=True ✓")
    except TypeError:
        opt = torch.optim.AdamW(groups, lr=cfg.lr_peak, betas=(0.9, 0.95), eps=1e-8)
        if master: print("  AdamW fused=False (upgrade PyTorch for fused)")

    if master:
        print(f"  Decay params   : {sum(p.numel() for p in decay):,}")
        print(f"  No-decay params: {sum(p.numel() for p in no_decay):,}")
    return opt

if master: print("OPTIMIZER")
optimizer = configure_optimizer(raw_model, train_cfg)

# AMP scaler — prevents fp16 gradient underflow
scaler = torch.amp.GradScaler("cuda", enabled=(device != "cpu"))


# ===========================================================================
# CELL 15 — Checkpoint Save / Load
# ===========================================================================

os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

def save_checkpoint(model, optimizer, scaler, step, loss, cfg: TrainConfig):
    # Only master saves to avoid file conflicts
    if not master:
        return
    path = os.path.join(cfg.checkpoint_dir, f"ckpt_{step:07d}.pt")
    torch.save({
        "step"            : step,
        "model_state"     : model.state_dict(),
        "optimizer_state" : optimizer.state_dict(),
        "scaler_state"    : scaler.state_dict(),
        "loss"            : loss,
        "model_config"    : {
            "d_model"    : model_cfg.d_model,
            "n_heads"    : model_cfg.n_heads,
            "n_layers"   : model_cfg.n_layers,
            "context_len": model_cfg.context_len,
            "vocab_size" : model_cfg.vocab_size,
        },
    }, path)
    print(f"  ✓ Saved → {path}")

def load_checkpoint(model, optimizer, scaler, path):
    if master: print(f"  Loading: {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    step = ckpt["step"]
    loss = ckpt["loss"]
    if master: print(f"  ✓ Resumed step={step}  loss={loss:.4f}")
    return step, loss

if master: print(f"CHECKPOINTS → {train_cfg.checkpoint_dir}")


# ===========================================================================
# CELL 16 — Evaluation + Generation helpers
# ===========================================================================

@torch.no_grad()
def evaluate(model, loader, device, num_batches=30):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= num_batches: break
        with torch.amp.autocast("cuda", enabled=(device != "cpu")):
            _, loss = model(x.to(device, non_blocking=True),
                            y.to(device, non_blocking=True))
        losses.append(loss.item())
    model.train()
    avg_loss   = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

@torch.no_grad()
def generate_sample(model, enc, prompt, max_new_tokens=150,
                    temperature=0.8, top_k=50, yarn_scale=1.0):
    model.eval()
    tokens = enc.encode(prompt)
    idx    = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    # Use raw_model for generation (DDP wrapper doesn't expose generate)
    _m     = model.module if hasattr(model, "module") else model
    out    = _m.generate(idx, max_new_tokens, temperature, top_k, yarn_scale)
    model.train()
    return enc.decode(out[0].tolist())


# ===========================================================================
# CELL 17 — TRAINING LOOP
# ===========================================================================

start_step = 0
if train_cfg.resume_from:
    start_step, _ = load_checkpoint(raw_model, optimizer, scaler, train_cfg.resume_from)

model.train()
train_iter = iter(train_loader)

if master:
    print("=" * 65)
    print("TRAINING 300M")
    print(f"  steps        : {train_cfg.max_steps:,}")
    print(f"  eff batch    : {train_cfg.batch_size * train_cfg.accum_steps}")
    print(f"  tokens/step  : {train_cfg.batch_size * train_cfg.accum_steps * model_cfg.context_len:,}")
    print(f"  AMP fp16     : {device != 'cpu'}")
    print("=" * 65)

t0 = time.time()

for step in range(start_step, train_cfg.max_steps):

    # Update LR
    lr = get_lr(step, train_cfg)
    for g in optimizer.param_groups:
        g["lr"] = lr

    # Gradient accumulation
    optimizer.zero_grad(set_to_none=True)
    step_loss = 0.0

    for micro_step in range(train_cfg.accum_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        # Move to GPU then immediately delete CPU tensors — halves RAM at this point
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device != "cpu")):
            _, loss = model(x, y)
        loss = loss / train_cfg.accum_steps
        scaler.scale(loss).backward()
        step_loss += loss.item()

        # Explicitly free CPU tensors and loss — prevents gradual RAM leak
        del x, y, loss

    # Unscale before clip so grad_norm is in real units
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

    scaler.step(optimizer)
    scaler.update()

    # Logging — only master prints
    if master and (step == 0 or step % train_cfg.log_interval == 0):
        t1        = time.time()
        elapsed   = t1 - t0
        t0        = t1
        n         = 1 if step == 0 else train_cfg.log_interval
        tok_s     = (n * train_cfg.batch_size *
                     train_cfg.accum_steps * model_cfg.context_len) / elapsed
        gpu_mem   = torch.cuda.memory_allocated() / 1e9 if device != "cpu" else 0
        print(f"step {step:7d} | loss {step_loss:.4f} | lr {lr:.2e} | "
              f"grad {grad_norm:.3f} | {tok_s/1e3:.1f}k tok/s | gpu {gpu_mem:.2f}GB")

    # Periodic GC — clears Python reference cycles that accumulate over steps
    if step % 500 == 0:
        gc.collect()

    # Eval
    if master and step % train_cfg.eval_interval == 0 and step > 0:
        val_loss, val_ppl = evaluate(model, val_loader, device)
        print(f"\n  [EVAL  {step:7d}] val_loss={val_loss:.4f}  perplexity={val_ppl:.2f}\n")

    # Sample
    if master and step % train_cfg.sample_interval == 0 and step > 0:
        out = generate_sample(model, enc, "The fundamental principles of",
                              max_new_tokens=120, temperature=0.8, top_k=50)
        print(f"\n  [SAMPLE {step:7d}]\n  {out}\n")

    # Checkpoint
    if master and step % train_cfg.save_interval == 0 and step > 0:
        save_checkpoint(raw_model, optimizer, scaler, step, step_loss, train_cfg)



# Final save
if master:
    save_checkpoint(raw_model, optimizer, scaler, train_cfg.max_steps, step_loss, train_cfg)
    final = generate_sample(model, enc, "Intelligence emerges from",
                            max_new_tokens=150, temperature=0.8, top_k=50)
    print(f"\n[FINAL SAMPLE]\n{final}\n")
    print("✓ Training complete!")




# ===========================================================================
# CELL 18 — PLATEAU RESUME CELL
# ===========================================================================
# Run this cell INSTEAD of Cell 17 when loss plateaus.
# Detects plateau, lets you override LR, resumes cleanly.
#
# How to detect plateau:
#   val loss flat for 3000+ steps  → plateau
#   grad_norm very low + flat      → plateau
#   train and val loss both stuck  → plateau
#
# DO NOT restart from scratch. Load checkpoint, rewarm LR, continue.

# --- Plateau Resume Config ---
PLATEAU_RESUME_PATH = "/kaggle/working/checkpoints_300M/ckpt_0040000.pt"  # ← update path

# Override LR for rewarm — lower than original peak, higher than where you were
train_cfg.lr_peak     = 1e-4    # ← adjust based on where plateau hit
train_cfg.lr_min      = 1e-5
train_cfg.warmup_steps = 300    # short rewarm — just enough to stabilize
# Keep max_steps the same — cosine recalculates from loaded step number

# Load checkpoint
plateau_step, _ = load_checkpoint(raw_model, optimizer, scaler, PLATEAU_RESUME_PATH)

if master:
    print(f"Resuming from step {plateau_step} with new LR peak={train_cfg.lr_peak}")
    print(f"Rewarm for {train_cfg.warmup_steps} steps then cosine to step {train_cfg.max_steps}")

# Now re-run Cell 17 (training loop) — it reads train_cfg for LR
# The loaded step_number ensures cosine schedule continues from right position
# start_step will be set to plateau_step automatically via resume_from


# ===========================================================================
# CELL 19 — YaRN Inference (long context beyond 2048)
# ===========================================================================
# At inference, extend context window beyond training length using YaRN.
# No retraining needed — just scale RoPE frequencies.
#
# Recommended scales:
#   yarn_scale=1.0  → 2048  (training length, perfect)
#   yarn_scale=2.0  → 4096  (excellent quality)
#   yarn_scale=4.0  → 8192  (good quality)
#   yarn_scale=8.0  → 16384 (decent, some degradation)

def generate_long(prompt, max_new_tokens=500, yarn_scale=4.0, temperature=0.7, top_k=50):
    """Generate with extended context via YaRN scaling."""
    effective_ctx = int(model_cfg.context_len * yarn_scale)
    if master:
        print(f"YaRN scale={yarn_scale} → effective context={effective_ctx} tokens")
    return generate_sample(
        model, enc, prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        yarn_scale=yarn_scale,
    )

# Example usage after training:
# out = generate_long("The history of artificial intelligence began", yarn_scale=4.0)
# print(out)


# ===========================================================================
# CELL 20 — LoRA Finetuning Setup (run after pretraining)
# ===========================================================================
# Switch model to finetuning mode:
#   1. Set use_lora=True in ModelConfig
#   2. Rebuild model (LoRALinear replaces nn.Linear in QKV/Out/MLP)
#   3. Load pretrained weights into base layers
#   4. Freeze base weights
#   5. Train on instruction data — only LoRA A and B matrices update

def prepare_for_finetuning(pretrained_ckpt_path):
    # Build new model with LoRA enabled
    ft_cfg         = ModelConfig()
    ft_cfg.use_lora = True
    ft_model        = GPT(ft_cfg).to(device)

    # Load pretrained base weights (strict=False — LoRA params are new)
    ckpt = torch.load(pretrained_ckpt_path, map_location=device)
    missing, unexpected = ft_model.load_state_dict(ckpt["model_state"], strict=False)

    if master:
        print(f"  Missing keys (LoRA params, expected): {len(missing)}")
        print(f"  Unexpected keys                      : {len(unexpected)}")

    # Freeze all base weights — only LoRA A and B train
    ft_model.freeze_base_weights()

    return ft_model

# Usage:
# ft_model  = prepare_for_finetuning("/kaggle/working/checkpoints_300M/ckpt_0120000.pt")
# optimizer = configure_optimizer(ft_model, train_cfg)
# Then train ft_model on OpenHermes / LIMA instruction data using same training loop


# ===========================================================================
# CELL 21 — Scale-Up Reference (no code changes needed)
# ===========================================================================
# To scale, ONLY change ModelConfig. Every other cell is identical.
#
# 300M (current):
#   d_model=1024, n_heads=16, n_layers=18, context_len=2048
#   Kaggle 2×T4, ~30hrs, 8B tokens
#
# 500M:
#   d_model=1024, n_heads=16, n_layers=30
#   Needs ~10B tokens, ~55hrs on 2×T4 (2 Kaggle accounts)
#
# 1B:
#   d_model=2048, n_heads=16, n_layers=18
#   Needs ~20B tokens, A100 recommended
#
# Run command for DDP (2 GPUs):
#   torchrun --nproc_per_node=2 train.py
#
# Single GPU (Colab):
#   python train.py   (DDP auto-disabled, LOCAL_RANK not set)

if master:
    print("All cells loaded. Ready to train.")
    print("For DDP: torchrun --nproc_per_node=2 this_file.py")
    print("For single GPU: python this_file.py")
