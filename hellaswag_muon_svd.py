import os, time, uuid, math, json, argparse, random
from pathlib import Path
from typing import Iterable, Callable, Dict, List, Tuple

import numpy as np
import torch
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.amp import GradScaler

from datasets import load_dataset
from transformers import GPT2TokenizerFast

try:
    import torch._inductor.config as ic
    if hasattr(ic, "triton") and hasattr(ic.triton, "cudagraph_skip_dynamic_graphs"):
        ic.triton.cudagraph_skip_dynamic_graphs = True
        ic.triton.cudagraph_dynamic_shape_warn_limit = None
except Exception:
    pass  # Ignore if not available

def setup_distributed() -> Tuple[bool, int, int, int]:
    """
    Initialize torch.distributed if WORLD_SIZE > 1.
    Returns: (distributed, rank, local_rank, world_size)

    - On GPU + WORLD_SIZE>1: sets up NCCL DDP.
    - On CPU: only single-process is supported; WORLD_SIZE must be 1.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # CPU-only mode: allow single process, disallow multi-process DDP
    if not torch.cuda.is_available():
        if world_size > 1:
            raise RuntimeError(
                "WORLD_SIZE>1 but no CUDA device is available. "
                "Multi-process training currently requires GPUs (NCCL backend)."
            )
        return False, 0, 0, 1

    # GPU available
    if world_size > 1:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world_size
    else:
        return False, 0, 0, 1

    

# ========================================================================== #
# ───────────────────────────────  Muon  ─────────────────────────────────── #
# ========================================================================== #


def NewtonSchulz(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """
    Approximate an orthogonalized / polar-like transform of G via a fixed
    polynomial Newton–Schulz iteration.

    - Expects a 2D tensor (m x n).
    - Returns the same shape & dtype as the input.
    """
    if G.ndim != 2:
        raise ValueError(f"NewtonSchulz expects a 2D tensor, got {G.ndim}D")

    orig_dtype = G.dtype
    frob_norm = G.norm()

    # Tiny / zero-norm: just return zeros to avoid numerical issues.
    if frob_norm <= eps:
        return torch.zeros_like(G)

    if steps == 0:
        return (G / (frob_norm + eps)).to(orig_dtype)

    # Choose compute dtype:
    # - On CUDA, use bf16 *only* if the device supports it.
    # - Otherwise stick to float32.
    if G.is_cuda:
        try:
            use_bf16 = torch.cuda.is_bf16_supported()
        except AttributeError:
            use_bf16 = False
        comp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    else:
        comp_dtype = torch.float32

    X = (G / frob_norm).to(comp_dtype)

    m, n = X.shape
    transposed = False
    if m > n:
        X = X.T
        transposed = True

    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X

    if transposed:
        X = X.T

    return X.to(orig_dtype)



def PolarSVD(G: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Exact polar orthogonal factor via SVD.
    Given 2D matrix G (m x n), returns Q = U @ V^T where G = U Σ V^T.

    Shape and dtype are preserved.
    """
    if G.ndim != 2:
        raise ValueError(f"PolarSVD expects a 2D tensor, got {G.ndim}D")

    if G.norm() <= eps:
        return torch.zeros_like(G)

    orig_dtype = G.dtype
    X = (G.to(torch.float32)) / (G.norm() + eps)

    try:
        U, _, Vh = torch.linalg.svd(X, full_matrices=False)
    except RuntimeError:
        # Fallback to CPU if GPU SVD is unhappy, then move back.
        U, _, Vh = torch.linalg.svd(X.cpu(), full_matrices=False)
        U, Vh = U.to(G.device), Vh.to(G.device)

    Q = U @ Vh
    return Q.to(orig_dtype)


class Muon(optim.Optimizer):
    """
    Muon: SGD with momentum + Newton–Schulz low-rank update + unit-norm re-scaling.

    Intended use:
      - Apply to matrix-like parameters (ndim >= 2).
      - 1D parameters can be handled by a separate small SGD optimizer
        (as in your script), but this class gracefully falls back to a
        plain momentum step for 1D tensors.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        steps: int = 5,
        eps: float = 1e-12,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if steps < 0:
            raise ValueError(f"steps must be >= 0, got {steps}")

        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, steps=steps, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            mom: float = group["momentum"]
            nesterov: bool = group["nesterov"]
            steps: int = group["steps"]
            eps: float = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                state = self.state.setdefault(p, {})
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["momentum_buffer"] = buf

                buf.mul_(mom).add_(grad)
                grad_hat = grad.add(buf, alpha=mom) if nesterov else buf

                # Matrix-like params: rescale + Newton–Schulz update
                if p.ndim >= 2:
                    rows = p.shape[0]
                    target_norm = math.sqrt(float(rows))

                    # Re-scale weight to fixed Frobenius norm ~ sqrt(rows)
                    p.mul_(target_norm / (p.norm() + eps))

                    G2 = grad_hat.reshape(rows, -1)
                    update = NewtonSchulz(G2, steps=steps).view_as(grad_hat)
                else:
                    # 1D params: simple momentum SGD step
                    update = grad_hat

                p.add_(update, alpha=-lr)

        return loss


class MuonSVD(optim.Optimizer):
    """
    MuonSVD: momentum SGD + exact polar (SVD) orthogonalization + unit-norm re-scaling.

    Same pattern as Muon but uses PolarSVD instead of Newton–Schulz.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        eps: float = 1e-12,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            mom: float = group["momentum"]
            nesterov: bool = group["nesterov"]
            eps: float = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("MuonSVD does not support sparse gradients")

                state = self.state.setdefault(p, {})
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["momentum_buffer"] = buf

                buf.mul_(mom).add_(grad)
                grad_hat = grad.add(buf, alpha=mom) if nesterov else buf

                if p.ndim >= 2:
                    rows = p.shape[0]
                    target_norm = math.sqrt(float(rows))

                    # Re-scale weight to fixed Frobenius norm ~ sqrt(rows)
                    p.mul_(target_norm / (p.norm() + eps))

                    G2 = grad_hat.reshape(rows, -1)
                    update = PolarSVD(G2).view_as(grad_hat)
                else:
                    update = grad_hat

                p.add_(update, alpha=-lr)

        return loss


# ========================================================================== #
# ───────────────────── HellaSwag dataset & collator ────────────────────── #
# ========================================================================== #

class HellaSwagTextDataset(Dataset):
    """
    Returns (context, endings[4], label) per example.
    We'll score each ending with the LM and treat it as 4‑way classification.
    """
    def __init__(self, split, max_examples: int | None = None):
        ds = load_dataset("Rowan/hellaswag", "default", split=split)
        if max_examples is not None:
            # avoid IndexError when max_examples > len(ds)
            max_examples = min(max_examples, len(ds))
            ds = ds.select(range(max_examples))
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        ex = self.ds[int(idx)]
        ctx_a = ex.get("ctx_a", "").strip()
        ctx_b = ex.get("ctx_b", "").strip()
        ctx = ex.get("ctx", "")  # in case you ever precompute & store it
        context = (ctx or (ctx_a + " " + ctx_b)).strip()
        endings = [e.strip() for e in ex["endings"]]
        label = int(ex["label"])
        return {
            "context": context,
            "endings": endings,
            "label": label,
        }


class HellaSwagCollator:
    """
    Tokenizes HellaSwag batch into:
      input_ids:  (B, 4, T)
      attention:  (B, 4, T)
      labels:     (B,)
    We do LM scoring over full sequence for each of the 4 endings.
    """
    def __init__(self, tokenizer: GPT2TokenizerFast, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __call__(self, batch):
        B = len(batch)
        texts: List[str] = []
        labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)
        for ex in batch:
            ctx = ex["context"]
            for end in ex["endings"]:
                texts.append(ctx + " " + end)

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.block_size,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]         # (B*4, T)
        attention = enc["attention_mask"]    # (B*4, T)

        num_choices = 4
        T = input_ids.size(1)
        input_ids = input_ids.view(B, num_choices, T)
        attention = attention.view(B, num_choices, T)
        return input_ids, attention, labels


def get_hellaswag_loaders(
    block_size: int = 256,
    batch_size: int = 8,
    tokenizer_name: str = "gpt2",
    max_train_examples: int | None = None,
    max_eval_examples: int | None = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, GPT2TokenizerFast]:
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)

    # GPT-2 has no pad token by default → use EOS as pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_ds = HellaSwagTextDataset("train", max_examples=max_train_examples)
    val_ds   = HellaSwagTextDataset("validation", max_examples=max_eval_examples)

    collator = HellaSwagCollator(tokenizer, block_size)

    if distributed and world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
    else:
        train_sampler = None
        val_sampler = None

    # Common kwargs for both loaders
    common_loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    # prefetch_factor is only valid when num_workers > 0
    if num_workers > 0:
        common_loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        **common_loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        **common_loader_kwargs,
    )

    return train_loader, val_loader, tokenizer


# ========================================================================== #
# ───────────────────── GPT‑2‑XL–style transformer ──────────────────────── #
# ========================================================================== #

class GPTConfig:
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int = 48,
        n_head: int = 25,
        n_embd: int = 1600,
        dropout: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.resid_drop = nn.Dropout(config.dropout)

        # keep dropout prob here, but use SDPA's fused dropout for speed
        self.attn_dropout_p = config.dropout

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, h, T, d)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Uses FlashAttention / SDPA when available (PyTorch 2.x, H100-friendly)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=True,
        )  # (B, h, T, d)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.resid_drop(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx: Tensor) -> Tensor:
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(f"Sequence length {T} > block_size {self.config.block_size}")
        pos = torch.arange(0, T, device=idx.device, dtype=torch.long).unsqueeze(0)

        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(pos)
        x = self.drop(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


def compiled_gpt(config: GPTConfig) -> nn.Module:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(config).to(device)
    try:
        return torch.compile(model, mode="reduce-overhead")
    except Exception:
        return model


def estimate_num_params(config: GPTConfig) -> int:
    """
    Analytic param count (avoids allocating a full 1.5B model just to count).
    Matches GPT-2-style architecture.
    """
    v = config.vocab_size
    d = config.n_embd
    L = config.n_layer
    T = config.block_size

    tok_emb = v * d
    pos_emb = T * d

    # per-block params:
    c_attn_w = d * (3 * d)
    c_attn_b = 3 * d
    c_proj_w = d * d
    c_proj_b = d
    c_fc_w = d * (4 * d)
    c_fc_b = 4 * d
    c_mlp_proj_w = (4 * d) * d
    c_mlp_proj_b = d
    ln = 2 * (2 * d)   # two LayerNorms per block

    block = c_attn_w + c_attn_b + c_proj_w + c_proj_b + c_fc_w + c_fc_b + c_mlp_proj_w + c_mlp_proj_b + ln
    ln_f = 2 * d

    return tok_emb + pos_emb + L * block + ln_f   # lm_head tied


# ========================================================================== #
# ────────────────── HellaSwag forward / evaluation logic ───────────────── #
# ========================================================================== #

def hellaswag_forward(
    model: nn.Module,
    input_ids: torch.Tensor,        # (B, 4, T)
    attention: torch.Tensor,        # (B, 4, T)
    labels: torch.Tensor | None,    # (B,)
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute multiple-choice loss & predictions:

    1. Run LM on all 4 endings per example.
    2. Compute mean NLL per ending.
    3. Convert to 4-way scores (higher is better).
    4. Cross-entropy over choices.
    """
    device = next(model.parameters()).device
    B, C, T = input_ids.shape
    input_ids = input_ids.view(B * C, T).to(device, non_blocking=True)
    attention = attention.view(B * C, T).to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True) if labels is not None else None

    is_cuda = device.type == "cuda"

    if is_cuda and use_amp:
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            logits = model(input_ids)
    else:
        logits = model(input_ids)

    # Shift for LM loss
    V = logits.size(-1)
    shift_logits = logits[:, :-1, :]           # (B*C, T-1, V)
    shift_labels = input_ids[:, 1:]            # (B*C, T-1)
    shift_mask   = attention[:, 1:]            # (B*C, T-1)

    loss_tokens = F.cross_entropy(
        shift_logits.reshape(-1, V),
        shift_labels.reshape(-1),
        reduction="none",
    )
    loss_tokens = loss_tokens.view(B * C, -1)
    loss_tokens = loss_tokens * shift_mask

    token_counts = shift_mask.sum(dim=1) + 1e-9
    seq_nll = loss_tokens.sum(dim=1) / token_counts          # (B*C,)
    seq_nll = seq_nll.view(B, C)                             # (B, 4)
    scores = -seq_nll                                        # higher is better

    if labels is None:
        preds = scores.argmax(dim=-1)
        return torch.tensor(0.0, device=device), preds

    loss = F.cross_entropy(scores, labels)
    preds = scores.argmax(dim=-1)
    return loss, preds


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    distributed: bool = False,
    world_size: int = 1,
) -> Tuple[float, float]:
    """
    Evaluate classification loss and accuracy on HellaSwag.
    Works for single-GPU and DDP (aggregates via all_reduce).
    """
    dev = device if isinstance(device, torch.device) else torch.device(device)
    is_cuda = dev.type == "cuda"

    model.eval()
    total_loss = torch.zeros(1, device=dev)
    total_correct = torch.zeros(1, device=dev)
    total_examples = torch.zeros(1, device=dev)

    with torch.inference_mode():
        for input_ids, attention, labels in loader:
            # input_ids = input_ids.to(dev, non_blocking=True)
            # attention = attention.to(dev, non_blocking=True)
            # labels = labels.to(dev, non_blocking=True)

            loss, preds = hellaswag_forward(
                model, input_ids, attention, labels,
                use_amp=is_cuda and use_amp,
                amp_dtype=amp_dtype,
            )

            bs = labels.size(0)
            labels_dev = labels.to(preds.device)
            total_loss += loss.detach() * bs
            total_correct += (preds == labels_dev).sum()
            total_examples += bs

    if distributed and world_size > 1 and dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_examples, op=dist.ReduceOp.SUM)

    avg_loss = (total_loss / total_examples.clamp_min(1.0)).item()
    acc = (total_correct / total_examples.clamp_min(1.0)).item()
    return avg_loss, acc



# ========================================================================== #
# ───────────────────────── Optimizers & schedulers ─────────────────────── #
# ========================================================================== #

def _make_sgd(
    param_groups,
    lr: float,
    momentum: float,
    nesterov: bool = True,
) -> optim.SGD:
    try:
        return optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            fused=True,
        )
    except TypeError:
        return optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
        )


def build_optimizers(
    name: str,
    model: nn.Module,
    hparams: dict | None = None
) -> list[optim.Optimizer]:
    """
    Build Muon / MuonSVD / SGD optimizers (same pattern as your CIFAR script).
    - Matrix-like params (ndim >= 2): main optimizer.
    - 1D params: small SGD optimizer.
    """
    hp = hparams or {}

    matrix_like, others, seen = [], [], set()
    for p in model.parameters():
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        if p.ndim >= 2:
            matrix_like.append(p)
        else:
            others.append(p)

    lr_other  = hp.get('lr_other', 0.001)
    mom_other = hp.get('momentum_other', 0.99)

    small_groups = []
    if others:
        small_groups.append(
            {"params": others, "lr": lr_other, "momentum": mom_other, "nesterov": True}
        )

    opt_small = _make_sgd(
        small_groups, lr=lr_other, momentum=mom_other, nesterov=True
    ) if small_groups else None

    lr_main = hp.get("lr", {
        "muon":     0.02,
        "muon_svd": 0.02,
        "sgd":      0.05,
    }[name])

    if name == "muon":
        opt_main = Muon(
            [{"params": matrix_like, "lr": lr_main}],
            lr=lr_main,
            momentum=hp.get("momentum", 0.95),
            nesterov=hp.get("nesterov", True),
            steps=hp.get("steps", 5),
        )
    elif name == "muon_svd":
        opt_main = MuonSVD(
            [{"params": matrix_like, "lr": lr_main}],
            lr=lr_main,
            momentum=hp.get("momentum", 0.95),
            nesterov=hp.get("nesterov", True),
        )
    elif name == "sgd":
        opt_main = _make_sgd(
            [{"params": matrix_like, "lr": lr_main}],
            lr=lr_main,
            momentum=hp.get("momentum", 0.9),
            nesterov=hp.get("nesterov", True),
        )
    else:
        raise ValueError(f"Unknown optimizer '{name}'")

    return [opt for opt in (opt_small, opt_main) if opt is not None]


def build_schedulers(
    optimizers: list[optim.Optimizer],
    steps_per_epoch: int,
    epochs: int,
    warmup_ratio: float = 0.05,
) -> list[optim.lr_scheduler.LambdaLR]:
    """Warm‑up + cosine LR schedule applied to all optimizers (per-step)."""
    total_steps = steps_per_epoch * epochs
    warm_steps  = int(total_steps * warmup_ratio)

    def schedule(step: int, w=warm_steps, t=total_steps) -> float:
        """
        - step is 0-based internal step counter from LambdaLR.
        - We use s = step + 1 so that the very first call does not produce 0 LR.
        """
        if t <= 0:
            return 1.0

        s = step + 1  # 1..total_steps

        if w <= 0:
            warm = 1.0
        else:
            warm = min(1.0, s / w)

        if t <= w:
            cosine = 1.0
        else:
            cosine = 0.5 * (1 + math.cos(math.pi * max(0, s - w) / max(1, t - w)))

        return warm * cosine

    return [optim.lr_scheduler.LambdaLR(opt, lr_lambda=schedule) for opt in optimizers]


def run_experiment(
    opt_name: str,
    runs: int,
    epochs: int,
    model_fn: Callable[[], nn.Module],
    hparams: dict | None = None,
    batch_size: int = 8,
    block_size: int = 256,
    tokenizer_name: str = "gpt2",
    max_train_examples: int | None = None,
    max_eval_examples: int | None = None,
    log_interval: int = 0,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 4,
    warmup_ratio: float = 0.1,
    use_compile: bool = True,
) -> dict[str, List[List[float]]]:
    """
    Train a single optimizer (multiple runs) on HellaSwag + GPT model.

    - Supports single GPU or DDP.
    - By default, only end-of-epoch validation (no mid-epoch full eval).
    """
    results = {
        "train":      [],   # train loss per epoch
        "val_loss":   [],   # validation loss per epoch
        "acc":        [],   # validation accuracy per epoch
        "time":       [],   # cumulative seconds per epoch
        "train_step": [],
        "acc_step":   [],
        "time_step":  [],
    }
    hp = hparams or {}
    steps_str = hp.get("steps", "N/A")
    if rank == 0:
        print(f"\n── {opt_name.upper()} (steps={steps_str}) ──")
        print(" | ".join(f"{c:^10}" for c in ("run", "epoch", "train", "acc", "sec")))

    for run in range(1, runs + 1):
        # Same seeds across ranks for DDP; vary by run
        torch.manual_seed(run)
        np.random.seed(run)
        random.seed(run)

        # Data loaders (per run so samplers reset cleanly)
        train_loader, val_loader, _ = get_hellaswag_loaders(
            block_size=block_size,
            batch_size=batch_size,
            tokenizer_name=tokenizer_name,
            max_train_examples=max_train_examples,
            max_eval_examples=max_eval_examples,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Build model
        model = model_fn()
        if torch.cuda.is_available():
            device = torch.device(torch.cuda.current_device())
            model = model.to(device)
            is_cuda = True
            use_bf16 = torch.cuda.is_bf16_supported()
            amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
            use_amp = True
        else:
            device = torch.device("cpu")
            model = model.to(device)
            is_cuda = False
            amp_dtype = torch.float32
            use_amp = False
            
        # Use GradScaler only for fp16 (not needed / useful for bf16)
        scaler = GradScaler(enabled=(is_cuda and use_amp and amp_dtype == torch.float16))

        # Compile before wrapping in DDP
        if is_cuda and use_compile:
            try:
                model = torch.compile(model, mode="max-autotune")
            except Exception:
                if rank == 0:
                    print("⚠️ torch.compile failed; continuing without compilation.")

        if distributed and is_cuda and world_size > 1:
            model = DDP(
                model,
                device_ids=[torch.cuda.current_device()],
                output_device=torch.cuda.current_device(),
                gradient_as_bucket_view=True,
            )

        optimizers = build_optimizers(opt_name, model, hparams)
        schedulers = build_schedulers(
            optimizers,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            warmup_ratio=warmup_ratio,
        )

        # Per-run logs
        run_loss_epoch, run_val_loss_epoch, run_acc_epoch, run_sec_epoch = [], [], [], []
        run_loss_step, run_acc_step, run_sec_step = [], [], []

        t0 = time.perf_counter()
        global_step = 0

        for epoch in range(1, epochs + 1):
            model.train()

            # Local running stats for logging on this rank
            epoch_loss = 0.0
            nb = 0

            # Ensure different shuffling per epoch in DDP
            if distributed and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            for batch_idx, (input_ids, attention, labels) in enumerate(train_loader, start=1):
                loss, _ = hellaswag_forward(
                    model, input_ids, attention, labels,
                    use_amp=is_cuda and use_amp,
                    amp_dtype=amp_dtype,
                )

                # --- backward + optimizer step (unchanged) ---
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    for opt in optimizers:
                        scaler.unscale_(opt)
                    clip_grad_norm_(model.parameters(), 1.0)

                    for opt, sch in zip(optimizers, schedulers):
                        scaler.step(opt)
                        sch.step()
                    scaler.update()
                else:
                    loss.backward()
                    clip_grad_norm_(model.parameters(), 1.0)

                    for opt, sch in zip(optimizers, schedulers):
                        opt.step()
                        sch.step()

                model.zero_grad(set_to_none=True)
                # ----------------------------------------------

                # --- Make per-step loss global across all GPUs ---
                # loss is a scalar tensor (mean CE over the local batch)
                loss_value = loss.detach()

                if distributed and world_size > 1 and dist.is_initialized():
                    # After this, loss_value is the average of all ranks' losses
                    dist.all_reduce(loss_value, op=dist.ReduceOp.AVG)

                step_loss = loss_value.item()

                # --- Epoch-level bookkeeping (now using global step_loss) ---
                epoch_loss += step_loss
                nb += 1
                global_step += 1

                # --- Optional per-step logging (rank 0 only) ---
                if log_interval > 0 and (global_step % log_interval == 0) and rank == 0:
                    avg_train_loss = epoch_loss / max(1, nb)
                    elapsed_mid = time.perf_counter() - t0

                    # Store *global* per-step loss and time
                    run_loss_step.append(step_loss)
                    run_acc_step.append(None)  # still no step-level accuracy
                    run_sec_step.append(elapsed_mid)

                    print(
                        f"run={run} "
                        f"epoch={epoch} "
                        f"step={batch_idx}/{len(train_loader)} "
                        f"global_step={global_step} "
                        f"loss={step_loss:.4f} "
                        f"avg_loss={avg_train_loss:.4f} "
                        f"time={elapsed_mid:.1f}s"
                    )

            # ---- End of epoch: GLOBAL epoch train loss ----
            epoch_loss /= max(1, nb)

            # Validation (already global inside evaluate)
            val_loss, val_acc = evaluate(
                model,
                val_loader,
                device,
                use_amp=is_cuda and use_amp,
                amp_dtype=amp_dtype,
                distributed=distributed,
                world_size=world_size,
            )
            elapsed = time.perf_counter() - t0

            run_loss_epoch.append(epoch_loss)
            run_val_loss_epoch.append(val_loss)
            run_acc_epoch.append(val_acc)
            run_sec_epoch.append(elapsed)

            if rank == 0:
                print(" | ".join(
                    f"{v:^10.4f}" if isinstance(v, float) else f"{v:^10}"
                    for v in (run, epoch, epoch_loss, val_acc, elapsed)
                ))


        # Store this run's results (only really used on rank 0)
        results["train"].append(run_loss_epoch)
        results["val_loss"].append(run_val_loss_epoch)
        results["acc"].append(run_acc_epoch)
        results["time"].append(run_sec_epoch)
        results["train_step"].append(run_loss_step)
        results["acc_step"].append(run_acc_step)
        results["time_step"].append(run_sec_step)   

        if rank == 0:
            print("-" * 60)

    return results


# ====================================================================== #
# Main: run experiments and save JSON
# ====================================================================== #

def build_gpt_config_from_args(args, vocab_size: int) -> GPTConfig:
    return GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare Muon / Muon-SVD / SGD-M on 1.5B GPT (GPT-2-XL-style) + HellaSwag."
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    # Experiment config
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps", type=int, nargs="+", default=[],
                        help="Newton–Schulz steps to test for Muon.")
    parser.add_argument("--include_sgd", action="store_true", default=False)
    parser.add_argument("--include_svd", action="store_true", default=False)
    parser.add_argument("--logdir", type=str, default=None,
                        help="Directory to save results. Default: logs/hellaswag_1p5b_<uuid>")

    # Optimizer base hyperparams
    parser.add_argument("--base_lr", type=float, default=0.02)
    parser.add_argument("--base_momentum", type=float, default=0.95)
    parser.add_argument("--muon_lr", type=float, default=None)
    parser.add_argument("--sgd_lr", type=float, default=None)
    parser.add_argument("--muon_svd_lr", type=float, default=None)
    parser.add_argument("--sgd_epochs", type=int, default=3,
                        help="Epochs for SGD with Momentum only (default: 3).")

    # Scheduler
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio for LR scheduler (default: 0.1).")

    # Model hyperparams: GPT‑2‑XL‑ish
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--n_layer", type=int, default=48)
    parser.add_argument("--n_head", type=int, default=25)
    parser.add_argument("--n_embd", type=int, default=1600)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Debugging / small-model mode
    parser.add_argument(
        "--tiny_debug",
        action="store_true",
        help="Use a very small GPT config + smaller subsets, suitable for CPU / small GPUs."
    )
    
    # Data / logging
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Per-GPU batch size (global batch = batch_size * world_size).")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader workers per process.")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1,
        help="Log mid-epoch train loss every N steps (set to 0 to disable).",
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile (Inductor/Triton); run in eager mode only.",
    )

    # Data limits (subsample for debugging)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_eval_examples", type=int, default=None)

    args = parser.parse_args()

    use_compile = not args.no_compile
    # Optional tiny model for quick debugging
    if args.tiny_debug:
        # You can tweak these numbers as you like; these are very conservative.
        args.n_layer = 4
        args.n_head = 4
        args.n_embd = 256
        args.block_size = 64

        # Keep batch sizes and dataset sizes small for CPU / tiny GPU runs
        args.batch_size = min(args.batch_size, 4)
        if args.max_train_examples is None:
            args.max_train_examples = 1024
        if args.max_eval_examples is None:
            args.max_eval_examples = 1024

        print(
            "🔧 tiny_debug: using a small GPT config "
            f"(layers={args.n_layer}, heads={args.n_head}, emb={args.n_embd}, "
            f"block_size={args.block_size}, batch_size={args.batch_size})"
        )
        
    # Global perf flags
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    torch.manual_seed(42); np.random.seed(42); random.seed(42)

    # Distributed setup
    distributed, rank, local_rank, world_size = setup_distributed()

    logdir = Path(args.logdir or (Path("logs") / f"hellaswag_1p5b_{uuid.uuid4().hex[:8]}"))
    if rank == 0:
        logdir.mkdir(parents=True, exist_ok=True)

    # Build tokenizer & config (vocab_size taken from tokenizer for safety)
    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer_name)
    vocab_size = tokenizer.vocab_size
    gpt_config = build_gpt_config_from_args(args, vocab_size)
    n_params = estimate_num_params(gpt_config)
    if rank == 0:
        print(f"Model config: layers={gpt_config.n_layer}, heads={gpt_config.n_head}, emb={gpt_config.n_embd}")
        print(f"Estimated params: {n_params/1e9:.3f}B")
        print(f"World size: {world_size}, per-GPU batch size: {args.batch_size}")

    def make_model() -> nn.Module:
        # Return an unwrapped model; compile/DDP handled in run_experiment
        return GPT(gpt_config)

    # One small warm-up (short, on subset) to compile kernels
    if use_compile:
        # One small warm-up (short, on subset) to compile kernels
        if rank == 0:
            print("🔥 One-time warm-up on HellaSwag (small subset)...")
        _ = run_experiment(
            opt_name="muon",
            runs=1,
            epochs=1,
            model_fn=make_model,
            hparams={"steps": 5, "lr": args.muon_lr or args.base_lr},
            batch_size=min(4, args.batch_size),
            block_size=args.block_size,
            tokenizer_name=args.tokenizer_name,
            max_train_examples=512,
            max_eval_examples=512,
            log_interval=0,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            num_workers=max(2, args.num_workers),
            warmup_ratio=args.warmup_ratio,
            use_compile=True,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    else:
        if rank == 0:
            print("⚙️ Skipping warm-up because torch.compile is disabled.")


    # # Experiment configs: Muon steps, SGD-M, Muon-SVD
    # EXPERIMENT_CONFIGS = [
    #     {
    #         "name": f"Muon (steps={s})",
    #         "opt_name": "muon",
    #         "hparams": {
    #             "lr": args.muon_lr or args.base_lr,
    #             "momentum": args.base_momentum,
    #             "steps": s,
    #         },
    #     }
    #     for s in args.steps
    # ]

    # if args.include_sgd:
    #     EXPERIMENT_CONFIGS.append(
    #         {
    #             "name": "SGD with Momentum",
    #             "opt_name": "sgd",
    #             "hparams": {
    #                 "lr": args.sgd_lr or (args.base_lr * 2.5),
    #                 "momentum": args.base_momentum,
    #                 "nesterov": True,
    #             },
    #         }
    #     )

    if args.include_svd:
        EXPERIMENT_CONFIGS = [
            {
                "name": "Muon with SVD",
                "opt_name": "muon_svd",
                "hparams": {
                    "lr": args.muon_svd_lr or (args.base_lr * 0.5),
                    "momentum": args.base_momentum,
                    "nesterov": True,
                },
            }
        ]

    ALL_RESULTS: Dict[str, dict] = {}
    per_experiment_epochs: Dict[str, int] = {}
    per_experiment_runs: Dict[str, int] = {}

    for cfg in EXPERIMENT_CONFIGS:
        runs_this = args.runs
        epochs_this = args.sgd_epochs if cfg["opt_name"] == "sgd" else args.epochs

        per_experiment_runs[cfg["name"]] = runs_this
        per_experiment_epochs[cfg["name"]] = epochs_this

        res = run_experiment(
            opt_name=cfg["opt_name"],
            runs=runs_this,
            epochs=epochs_this,
            model_fn=make_model,
            hparams=cfg["hparams"],
            batch_size=args.batch_size,
            block_size=args.block_size,
            tokenizer_name=args.tokenizer_name,
            max_train_examples=args.max_train_examples,
            max_eval_examples=args.max_eval_examples,
            log_interval=args.log_interval,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            num_workers=args.num_workers,
            warmup_ratio=args.warmup_ratio,
            use_compile=use_compile,
        )

        if rank == 0:
            ALL_RESULTS[cfg["name"]] = res

    # Save results & metadata to JSON (rank 0 only)
    if rank == 0:
        res_path = logdir / "results.json"
        meta_path = logdir / "metadata.json"
        with open(res_path, "w") as f:
            json.dump(ALL_RESULTS, f)
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "dataset": "HellaSwag",
                    "model": "GPT-2-XL-style (≈1.5B)",
                    "runs_default": args.runs,
                    "epochs_default": args.epochs,
                    "sgd_epochs": args.sgd_epochs,
                    "base_lr": args.base_lr,
                    "base_momentum": args.base_momentum,
                    "muon_lr": args.muon_lr,
                    "sgd_lr": args.sgd_lr,
                    "muon_svd_lr": args.muon_svd_lr,
                    "steps": args.steps,
                    "include_sgd": args.include_sgd,
                    "include_svd": args.include_svd,
                    "batch_size": args.batch_size,
                    "block_size": args.block_size,
                    "vocab_size": vocab_size,
                    "n_layer": args.n_layer,
                    "n_head": args.n_head,
                    "n_embd": args.n_embd,
                    "dropout": args.dropout,
                    "tokenizer_name": args.tokenizer_name,
                    "per_experiment_epochs": per_experiment_epochs,
                    "per_experiment_runs": per_experiment_runs,
                    "estimated_params": n_params,
                    "max_train_examples": args.max_train_examples,
                    "max_eval_examples": args.max_eval_examples,
                    "warmup_ratio": args.warmup_ratio,
                    "world_size": world_size,
                },
                f,
                indent=2,
            )

        print(f"\n✅ Finished. Results saved to:\n  - {res_path}\n  - {meta_path}")

    # Clean up distributed
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()



if __name__ == "__main__":
    main()
