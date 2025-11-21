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


# ========================================================================== #
# ───────────────────── Distributed setup helper ─────────────────────────── #
# ========================================================================== #

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

    - Expects a 2D tensor (m x n) or batched (..., m, n).
    - Returns the same shape & dtype as the input.
    """
    if G.ndim < 2:
        raise ValueError(f"NewtonSchulz expects tensor with ndim>=2, got {G.ndim}")

    orig_dtype = G.dtype
    # We'll treat the last two dims as the matrix
    shape = G.shape
    m, n = shape[-2], shape[-1]
    X = G.reshape(-1, m, n)

    # Frobenius norm per matrix
    frob = X.norm(dim=(1, 2), keepdim=True)
    mask_small = frob <= eps
    frob = torch.where(mask_small, torch.ones_like(frob), frob)
    X = X / frob

    # Choose compute dtype
    if X.is_cuda:
        try:
            use_bf16 = torch.cuda.is_bf16_supported()
        except AttributeError:
            use_bf16 = False
        comp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    else:
        comp_dtype = torch.float32

    X = X.to(comp_dtype)

    # If m>n, work with transposed for efficiency
    transposed = m > n
    if transposed:
        X = X.transpose(1, 2)
        m, n = n, m

    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = X @ X.transpose(1, 2)  # (..., m, m)
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.transpose(1, 2)

    X = X.to(orig_dtype)
    X = X.reshape(*shape)
    # zero out tiny matrices
    X = torch.where(mask_small.reshape(-1, 1, 1), torch.zeros_like(X), X)
    return X


def PolarSVD(G: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Exact polar orthogonal factor via SVD.
    Given 2D matrix G (m x n) or batched (..., m x n), returns Q ~ U @ V^T.
    """
    if G.ndim < 2:
        raise ValueError(f"PolarSVD expects tensor with ndim>=2, got {G.ndim}")

    orig_dtype = G.dtype
    shape = G.shape
    m, n = shape[-2], shape[-1]
    X = G.reshape(-1, m, n)

    frob = X.norm(dim=(1, 2), keepdim=True)
    mask_small = frob <= eps
    frob = torch.where(mask_small, torch.ones_like(frob), frob)
    X = (X / frob).to(torch.float32)

    Q_list = []
    for mat in X:  # small batch loop; used only for reference Muon-SVD
        if mat.norm() <= eps:
            Q_list.append(torch.zeros_like(mat))
            continue
        try:
            U, _, Vh = torch.linalg.svd(mat, full_matrices=False)
        except RuntimeError:
            U, _, Vh = torch.linalg.svd(mat.cpu(), full_matrices=False)
            U, Vh = U.to(mat.device), Vh.to(mat.device)
        Q_list.append(U @ Vh)

    Q = torch.stack(Q_list, dim=0)
    Q = Q.to(orig_dtype)
    Q = torch.where(mask_small, torch.zeros_like(Q), Q)
    return Q.reshape(*shape)


class Muon(optim.Optimizer):
    """
    Simple (non-distributed) Muon: SGD with momentum + Newton–Schulz update.

    - Matrix-like params (ndim >= 2): apply Muon.
    - 0/1-D params: just momentum SGD.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        steps: int = 5,
        weight_decay: float = 0.0,
        eps: float = 1e-12,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if steps < 0:
            raise ValueError(f"steps must be >= 0, got {steps}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            steps=steps,
            weight_decay=weight_decay,
            eps=eps,
        )
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
            wd: float = group["weight_decay"]
            eps: float = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                if wd != 0.0 and p.ndim >= 2:
                    grad = grad.add(p, alpha=wd)

                state = self.state.setdefault(p, {})
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["momentum_buffer"] = buf

                buf.mul_(mom).add_(grad)
                grad_hat = grad.add(buf, alpha=mom) if nesterov else buf

                if p.ndim >= 2:
                    # Rescale to roughly fixed Frobenius norm ~ sqrt(rows)
                    rows = p.shape[0]
                    target_norm = math.sqrt(float(rows))
                    p_norm = p.norm()
                    if p_norm > 0:
                        p.mul_(target_norm / (p_norm + eps))

                    G2 = grad_hat.reshape(rows, -1)
                    G2 = G2.unsqueeze(0)  # batched
                    update = NewtonSchulz(G2, steps=steps).squeeze(0).view_as(grad_hat)
                else:
                    update = grad_hat

                p.add_(update, alpha=-lr)

        return loss


class MuonSVD(optim.Optimizer):
    """
    Reference Muon variant using exact SVD-based polar factor.
    Very slow compared to Newton–Schulz, but useful for comparisons.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        eps: float = 1e-12,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            eps=eps,
        )
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
            wd: float = group["weight_decay"]
            eps: float = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("MuonSVD does not support sparse gradients")

                if wd != 0.0 and p.ndim >= 2:
                    grad = grad.add(p, alpha=wd)

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
                    p_norm = p.norm()
                    if p_norm > 0:
                        p.mul_(target_norm / (p_norm + eps))

                    G2 = grad_hat.reshape(rows, -1)
                    G2 = G2.unsqueeze(0)
                    update = PolarSVD(G2).squeeze(0).view_as(grad_hat)
                else:
                    update = grad_hat

                p.add_(update, alpha=-lr)

        return loss


# ========================================================================== #
# ─────────────────────── FineWeb tokenization & data ────────────────────── #
# ========================================================================== #

def load_and_tokenize_fineweb(
    tokenizer: GPT2TokenizerFast,
    config_name: str = "sample-10BT",
    max_train_tokens: int = 10_000_000,
    max_val_tokens: int = 1_000_000,
    streaming: bool = True,
    seed: int = 42,
    rank: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stream a slice of FineWeb and tokenize it with GPT-2 BPE.

    We collect up to `max_train_tokens` for training and `max_val_tokens` for
    validation, in one pass through the (possibly gigantic) dataset.

    This is *much* smaller than the full FineWeb / sample-10BT split to keep
    things runnable on a single machine.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if streaming:
        ds = load_dataset(
            "HuggingFaceFW/fineweb",
            config_name,
            split="train",
            streaming=True,
        )
    else:
        ds = load_dataset(
            "HuggingFaceFW/fineweb",
            config_name,
            split="train",
        )

    train_ids: List[int] = []
    val_ids: List[int] = []

    random.seed(seed + rank)

    for ex in ds:
        text = ex["text"]
        if not text:
            continue
        ids = tokenizer.encode(text)
        # Add EOS between docs
        ids.append(tokenizer.eos_token_id)

        for tid in ids:
            if len(train_ids) < max_train_tokens:
                train_ids.append(tid)
            elif len(val_ids) < max_val_tokens:
                val_ids.append(tid)
            else:
                break

        if len(val_ids) >= max_val_tokens:
            break

    if len(train_ids) < max_train_tokens:
        print(
            f"⚠️ Only collected {len(train_ids)} train tokens "
            f"(requested {max_train_tokens})."
        )
    if len(val_ids) < max_val_tokens:
        print(
            f"⚠️ Only collected {len(val_ids)} validation tokens "
            f"(requested {max_val_tokens})."
        )

    train_tokens = torch.tensor(train_ids, dtype=torch.long)
    val_tokens = torch.tensor(val_ids, dtype=torch.long)
    return train_tokens, val_tokens


class TokenDataset(Dataset):
    """
    Simple contiguous token dataset for causal LM:
    Given a 1D tensor of token ids, returns blocks of length `block_size`
    with next-token targets.

      x = tokens[i*block_size : i*block_size + block_size]
      y = tokens[i*block_size + 1 : i*block_size + block_size + 1]
    """

    def __init__(self, tokens: torch.Tensor, block_size: int):
        assert tokens.ndim == 1
        self.tokens = tokens
        self.block_size = block_size
        self.num_blocks = (len(tokens) - 1) // block_size

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        i = int(idx)
        start = i * self.block_size
        end = start + self.block_size + 1
        x = self.tokens[start:end - 1]
        y = self.tokens[start + 1:end]
        return x, y


def get_fineweb_loaders(
    train_ds: Dataset,
    val_ds: Dataset,
    batch_size: int,
    distributed: bool,
    rank: int,
    world_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    if distributed and world_size > 1:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        train_sampler = None
        val_sampler = None

    common_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        common_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        **common_kwargs,
    )
    return train_loader, val_loader


# ========================================================================== #
# ─────────────────────── NanoGPT‑style GPT model ────────────────────────── #
# ========================================================================== #

class GPTConfig:
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        dropout: float = 0.0,
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

        self.attn_dropout_p = config.dropout

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, h, T, d)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=True,
        )  # (B, h, T, d)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
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

        tok_emb = self.tok_emb(idx)          # (B, T, C)
        pos_emb = self.pos_emb(pos)          # (1, T, C)
        x = self.drop(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)             # (B, T, V)
        return logits


def estimate_num_params(config: GPTConfig) -> int:
    """
    Rough analytic param count for this GPT architecture.
    """
    v = config.vocab_size
    d = config.n_embd
    L = config.n_layer
    T = config.block_size

    tok_emb = v * d
    pos_emb = T * d

    # per-block
    c_attn_w = d * (3 * d)
    c_attn_b = 3 * d
    c_proj_w = d * d
    c_proj_b = d
    c_fc_w = d * (4 * d)
    c_fc_b = 4 * d
    c_mlp_proj_w = (4 * d) * d
    c_mlp_proj_b = d
    ln = 2 * (2 * d)

    block = (
        c_attn_w + c_attn_b +
        c_proj_w + c_proj_b +
        c_fc_w + c_fc_b +
        c_mlp_proj_w + c_mlp_proj_b +
        ln
    )
    ln_f = 2 * d

    return tok_emb + pos_emb + L * block + ln_f


# ========================================================================== #
# ───────────────────── Language modeling forward & eval ─────────────────── #
# ========================================================================== #

def lm_forward(
    model: nn.Module,
    x: torch.Tensor,   # (B, T)
    y: torch.Tensor,   # (B, T)
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Standard causal LM forward: returns mean cross-entropy over all tokens.
    """
    device = next(model.parameters()).device
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    is_cuda = device.type == "cuda"

    if is_cuda and use_amp:
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            logits = model(x)
    else:
        logits = model(x)

    V = logits.size(-1)
    loss = F.cross_entropy(
        logits.view(-1, V),
        y.view(-1),
        reduction="mean",
    )
    return loss


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    distributed: bool = False,
    world_size: int = 1,
) -> tuple[float, float]:
    """
    Evaluate mean token-level cross-entropy and accuracy on validation set.

    Returns:
        (avg_loss, avg_accuracy)
    """
    dev = device if isinstance(device, torch.device) else torch.device(device)
    is_cuda = dev.type == "cuda"

    model.eval()
    total_loss = torch.zeros(1, device=dev)
    total_tokens = torch.zeros(1, device=dev)
    total_correct = torch.zeros(1, device=dev)

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            B, T = y.shape
            ntokens = B * T

            if is_cuda and use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits = model(x)
            else:
                logits = model(x)

            V = logits.size(-1)
            logits_flat = logits.view(-1, V)
            y_flat = y.view(-1)

            loss = F.cross_entropy(
                logits_flat,
                y_flat,
                reduction="sum",
            )

            preds = logits_flat.argmax(dim=-1)
            correct = (preds == y_flat).sum()

            total_loss += loss
            total_tokens += ntokens
            total_correct += correct

    if distributed and world_size > 1 and dist.is_initialized():
        for t in (total_loss, total_tokens, total_correct):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    denom = total_tokens.clamp_min(1)
    avg_loss = (total_loss / denom).item()
    avg_acc = (total_correct / denom).item()
    return avg_loss, avg_acc


# ========================================================================== #
# ───────────────────────── Optimizers & schedulers ─────────────────────── #
# ========================================================================== #

def _make_sgd(
    param_groups,
    lr: float,
    momentum: float,
    nesterov: bool = True,
    weight_decay: float = 0.0,
) -> optim.SGD:
    try:
        return optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            fused=True,
        )
    except TypeError:
        return optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
        )


def build_optimizers(
    name: str,
    model: nn.Module,
    hparams: dict | None = None,
) -> list[optim.Optimizer]:
    """
    Build Muon / MuonSVD / SGD optimizers.

    - Matrix-like params (ndim >= 2): main optimizer.
    - 0/1-D params: small SGD optimizer.
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

    lr_other = hp.get("lr_other", 0.001)
    mom_other = hp.get("momentum_other", 0.99)
    wd_other = hp.get("weight_decay_other", 0.0)

    small_groups = []
    if others:
        small_groups.append(
            dict(
                params=others,
                lr=lr_other,
                momentum=mom_other,
                nesterov=True,
                weight_decay=wd_other,
            )
        )

    opt_small = _make_sgd(
        small_groups,
        lr=lr_other,
        momentum=mom_other,
        nesterov=True,
        weight_decay=wd_other,
    ) if small_groups else None

    lr_main = hp.get("lr", {
        "muon":     0.02,
        "muon_svd": 0.02,
        "sgd":      0.05,
    }[name])
    mom_main = hp.get("momentum", 0.95)
    wd_main = hp.get("weight_decay", 0.0)
    nesterov = hp.get("nesterov", True)

    if name == "muon":
        opt_main = Muon(
            [{"params": matrix_like, "lr": lr_main}],
            lr=lr_main,
            momentum=mom_main,
            nesterov=nesterov,
            steps=hp.get("steps", 5),
            weight_decay=wd_main,
        )
    elif name == "muon_svd":
        opt_main = MuonSVD(
            [{"params": matrix_like, "lr": lr_main}],
            lr=lr_main,
            momentum=mom_main,
            nesterov=nesterov,
            weight_decay=wd_main,
        )
    elif name == "sgd":
        opt_main = _make_sgd(
            [{"params": matrix_like, "lr": lr_main}],
            lr=lr_main,
            momentum=mom_main,
            nesterov=nesterov,
            weight_decay=wd_main,
        )
    else:
        raise ValueError(f"Unknown optimizer '{name}'")

    return [opt for opt in (opt_small, opt_main) if opt is not None]


def build_schedulers(
    optimizers: list[optim.Optimizer],
    total_steps: int,
    warmup_ratio: float = 0.1,
) -> list[optim.lr_scheduler.LambdaLR]:
    """Warm‑up + cosine LR schedule applied to all optimizers (per-step)."""
    warm_steps = int(total_steps * warmup_ratio)

    def schedule(step: int, w=warm_steps, t=total_steps) -> float:
        if t <= 0:
            return 1.0
        s = step + 1  # 1..t

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


# ========================================================================== #
# ─────────────────────── Run NanoGPT + FineWeb runs ────────────────────── #
# ========================================================================== #

def run_experiment(
    opt_name: str,
    runs: int,
    max_steps: int,
    model_fn: Callable[[], nn.Module],
    train_ds: Dataset,
    val_ds: Dataset,
    hparams: dict | None = None,
    batch_size: int = 32,
    block_size: int = 1024,
    log_interval: int = 100,
    eval_interval: int = 200,
    warmup_ratio: float = 0.1,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 4,
    use_compile: bool = True,
) -> dict[str, List[List[float]]]:
    """
    Train a NanoGPT-style LM on FineWeb for a fixed number of steps,
    comparing a single optimizer choice.

    Returns per-run logs:
        - train_loss / train_step / train_tokens / train_time
        - val_loss   / val_acc   / val_step   / val_tokens / val_time
    """
    hp = hparams or {}

    results = {
        "train_loss":   [],
        "train_step":   [],
        "train_tokens": [],
        "train_time":   [],
        "val_loss":     [],
        "val_acc":      [],
        "val_step":     [],
        "val_tokens":   [],
        "val_time":     [],
    }

    if rank == 0:
        steps_str = hp.get("steps", "N/A")
        print(f"\n── {opt_name.upper()} (steps={steps_str}) ──")
        print(f"Logging train loss every step (stored), "
              f"val loss/acc every {eval_interval} steps.")

    for run in range(1, runs + 1):
        # Per-run RNG seeds (identical across ranks)
        torch.manual_seed(run)
        np.random.seed(run)
        random.seed(run)

        # Build loaders per run (so samplers reset cleanly)
        train_loader, val_loader = get_fineweb_loaders(
            train_ds,
            val_ds,
            batch_size=batch_size,
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
            total_steps=max_steps,
            warmup_ratio=warmup_ratio,
        )

        # Per-run logs
        run_train_loss, run_train_step, run_train_tokens, run_train_time = [], [], [], []
        run_val_loss, run_val_acc, run_val_step, run_val_tokens, run_val_time = [], [], [], [], []

        t0 = time.perf_counter()
        global_step = 0
        global_tokens = 0

        tokens_per_step_local = batch_size * block_size
        tokens_per_step_global = tokens_per_step_local * world_size

        # Train loop
        model.train()
        train_iter = iter(train_loader)
        epoch_idx = 0

        while global_step < max_steps:
            try:
                x, y = next(train_iter)
            except StopIteration:
                epoch_idx += 1
                if distributed and isinstance(train_loader.sampler, DistributedSampler):
                    train_loader.sampler.set_epoch(epoch_idx)
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            loss = lm_forward(
                model,
                x,
                y,
                use_amp=is_cuda and use_amp,
                amp_dtype=amp_dtype,
            )

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

            global_step += 1
            global_tokens += tokens_per_step_global
            elapsed = time.perf_counter() - t0

            if rank == 0:
                # record per-step training metrics
                run_train_loss.append(loss.item())
                run_train_step.append(global_step)
                run_train_tokens.append(global_tokens)
                run_train_time.append(elapsed)

                # optional console logging
                if log_interval > 0 and (global_step % log_interval == 0):
                    avg_step_time = elapsed / max(global_step, 1)
                    print(
                        f"run:{run} "
                        f"step:{global_step}/{max_steps} "
                        f"train_loss:{loss.item():.4f} "
                        f"time:{elapsed*1000:.0f}ms "
                        f"step_avg:{avg_step_time*1000:.2f}ms"
                    )

            # Periodic full validation
            if eval_interval > 0 and (global_step % eval_interval == 0 or global_step == max_steps):
                val_loss, val_acc = evaluate(
                    model,
                    val_loader,
                    device,
                    use_amp=is_cuda and use_amp,
                    amp_dtype=amp_dtype,
                    distributed=distributed,
                    world_size=world_size,
                )
                elapsed_val = time.perf_counter() - t0

                if rank == 0:
                    run_val_loss.append(val_loss)
                    run_val_acc.append(val_acc)
                    run_val_step.append(global_step)
                    run_val_tokens.append(global_tokens)
                    run_val_time.append(elapsed_val)

                    print(
                        f"[VAL] run:{run} "
                        f"step:{global_step}/{max_steps} "
                        f"train_loss:{loss.item():.4f} "
                        f"val_loss:{val_loss:.4f} "
                        f"val_acc:{val_acc:.4f} "
                        f"time:{elapsed_val*1000:.0f}ms"
                    )

                # go back to training mode after eval
                model.train()

        # Store per-run logs
        results["train_loss"].append(run_train_loss)
        results["train_step"].append(run_train_step)
        results["train_tokens"].append(run_train_tokens)
        results["train_time"].append(run_train_time)

        results["val_loss"].append(run_val_loss)
        results["val_acc"].append(run_val_acc)
        results["val_step"].append(run_val_step)
        results["val_tokens"].append(run_val_tokens)
        results["val_time"].append(run_val_time)

        if rank == 0:
            print("-" * 60)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    return results


# ========================================================================== #
# ──────────────────────────────── Main ──────────────────────────────────── #
# ========================================================================== #

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
        description="NanoGPT + FineWeb optimizer comparison: SGD-M vs Muon variants."
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)

    # Experiment config
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of random seeds per optimizer.")
    parser.add_argument("--max_steps", type=int, default=6000,
                        help="Training steps per run.")
    parser.add_argument("--eval_interval", type=int, default=250,
                        help="Evaluate on validation set every N steps.")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Log training loss every N steps.")
    parser.add_argument("--muon_steps", type=int, nargs="+", default=[1, 2, 3],
                        help="Newton–Schulz steps to test for Muon.")
    parser.add_argument("--logdir", type=str, default=None,
                        help="Directory to save results. Default: logs/fineweb_nanogpt_<uuid>")

    # Optimizer base hyperparams
    parser.add_argument("--base_lr", type=float, default=0.02)
    parser.add_argument("--base_momentum", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--muon_lr", type=float, default=None)
    parser.add_argument("--sgd_lr", type=float, default=None)
    parser.add_argument("--muon_svd_lr", type=float, default=None)

    # Scheduler
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio for LR scheduler.")

    # Model hyperparams: NanoGPT/GPT-2-small-ish
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--n_layer", type=int, default=24)
    parser.add_argument("--n_head", type=int, default=16)
    parser.add_argument("--n_embd", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Data / FineWeb
    parser.add_argument("--fineweb_config", type=str, default="sample-10BT",
                        help="FineWeb subset/config name (e.g. 'sample-10BT').")
    parser.add_argument("--max_train_tokens", type=int, default=10_000_000,
                        help="Max number of FineWeb tokens for training.")
    parser.add_argument("--max_val_tokens", type=int, default=1_000_000,
                        help="Max number of FineWeb tokens for validation.")
    parser.add_argument("--streaming", action="store_true", default=True,
                        help="Use streaming mode for FineWeb.")
    parser.add_argument("--no_streaming", action="store_true",
                        help="Disable streaming and load index into memory.")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Per-GPU batch size (number of sequences).")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers per process.")

    # Misc
    parser.add_argument("--no_compile", action="store_true",
                        help="Disable torch.compile on the model.")

    args = parser.parse_args()

    # Decide streaming flag
    if args.no_streaming:
        args.streaming = False

    # Global perf flags
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Distributed setup
    distributed, rank, local_rank, world_size = setup_distributed()

    logdir = Path(args.logdir or (Path("logs") / f"fineweb_nanogpt_{uuid.uuid4().hex[:8]}"))
    if rank == 0:
        logdir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer_name)
    vocab_size = tokenizer.vocab_size

    # FineWeb → token tensors
    if rank == 0:
        print(f"Loading & tokenizing FineWeb config='{args.fineweb_config}' "
              f"(train={args.max_train_tokens:,} tokens, "
              f"val={args.max_val_tokens:,} tokens)...")

    train_tokens, val_tokens = load_and_tokenize_fineweb(
        tokenizer=tokenizer,
        config_name=args.fineweb_config,
        max_train_tokens=args.max_train_tokens,
        max_val_tokens=args.max_val_tokens,
        streaming=args.streaming,
        seed=42,
        rank=rank,
    )

    if rank == 0:
        print(f"Collected {len(train_tokens):,} train tokens and "
              f"{len(val_tokens):,} val tokens.")

    # Build datasets
    train_ds = TokenDataset(train_tokens, block_size=args.block_size)
    val_ds = TokenDataset(val_tokens, block_size=args.block_size)

    if rank == 0:
        print(f"Train dataset blocks: {len(train_ds):,}")
        print(f"Val   dataset blocks: {len(val_ds):,}")

    # Model config
    gpt_config = build_gpt_config_from_args(args, vocab_size)
    n_params = estimate_num_params(gpt_config)

    if rank == 0:
        print(f"Model config: layers={gpt_config.n_layer}, "
              f"heads={gpt_config.n_head}, emb={gpt_config.n_embd}, "
              f"block={gpt_config.block_size}")
        print(f"Estimated params: {n_params/1e6:.2f}M")
        tokens_per_step = args.batch_size * args.block_size * world_size
        print(f"World size: {world_size}, per-GPU batch size: {args.batch_size}, "
              f"tokens/step (global): {tokens_per_step:,}")
        print(f"Max steps per run: {args.max_steps}, "
              f"approx tokens/run: {tokens_per_step * args.max_steps:,}")

    def make_model() -> nn.Module:
        return GPT(gpt_config)

    # Experiment configs: SGD-M, Muon-SVD, Muon (NS steps 1,2,3)
    EXPERIMENT_CONFIGS = []

    # SGD with momentum baseline
    EXPERIMENT_CONFIGS.append(
        {
            "name": "SGD with Momentum",
            "opt_name": "sgd",
            "hparams": {
                "lr": args.sgd_lr or (args.base_lr * 2.5),
                "momentum": args.base_momentum,
                "nesterov": True,
                "weight_decay": args.weight_decay,
            },
        }
    )

    # Muon with SVD
    EXPERIMENT_CONFIGS.append(
        {
            "name": "Muon with SVD",
            "opt_name": "muon_svd",
            "hparams": {
                "lr": args.muon_svd_lr or (args.base_lr * 0.5),
                "momentum": args.base_momentum,
                "nesterov": True,
                "weight_decay": args.weight_decay,
            },
        }
    )

    # Muon with Newton–Schulz steps
    for s in args.muon_steps:
        EXPERIMENT_CONFIGS.append(
            {
                "name": f"Muon (NS steps={s})",
                "opt_name": "muon",
                "hparams": {
                    "lr": args.muon_lr or args.base_lr,
                    "momentum": args.base_momentum,
                    "nesterov": True,
                    "weight_decay": args.weight_decay,
                    "steps": s,
                },
            }
        )

    ALL_RESULTS: Dict[str, dict] = {}

    for cfg in EXPERIMENT_CONFIGS:
        if rank == 0:
            print(f"\n=== Running experiment: {cfg['name']} ===")

        res = run_experiment(
            opt_name=cfg["opt_name"],
            runs=args.runs,
            max_steps=args.max_steps,
            model_fn=make_model,
            train_ds=train_ds,
            val_ds=val_ds,
            hparams=cfg["hparams"],
            batch_size=args.batch_size,
            block_size=args.block_size,
            log_interval=args.log_interval,
            eval_interval=args.eval_interval,
            warmup_ratio=args.warmup_ratio,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            num_workers=args.num_workers,
            use_compile=not args.no_compile,
        )

        if rank == 0:
            ALL_RESULTS[cfg["name"]] = res

    # Save results & metadata (rank 0 only)
    if rank == 0:
        res_path = logdir / "results.json"
        meta_path = logdir / "metadata.json"
        with open(res_path, "w") as f:
            json.dump(ALL_RESULTS, f)
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "dataset": "FineWeb",
                    "fineweb_config": args.fineweb_config,
                    "task": "NanoGPT-style causal LM",
                    "model": "NanoGPT-style GPT-2-small-ish",
                    "runs": args.runs,
                    "max_steps": args.max_steps,
                    "eval_interval": args.eval_interval,
                    "log_interval": args.log_interval,
                    "base_lr": args.base_lr,
                    "base_momentum": args.base_momentum,
                    "weight_decay": args.weight_decay,
                    "muon_lr": args.muon_lr,
                    "sgd_lr": args.sgd_lr,
                    "muon_svd_lr": args.muon_svd_lr,
                    "muon_steps": args.muon_steps,
                    "batch_size": args.batch_size,
                    "block_size": args.block_size,
                    "vocab_size": vocab_size,
                    "n_layer": args.n_layer,
                    "n_head": args.n_head,
                    "n_embd": args.n_embd,
                    "dropout": args.dropout,
                    "tokenizer_name": args.tokenizer_name,
                    "max_train_tokens": args.max_train_tokens,
                    "max_val_tokens": args.max_val_tokens,
                    "warmup_ratio": args.warmup_ratio,
                    "world_size": world_size,
                    "estimated_params": n_params,
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
