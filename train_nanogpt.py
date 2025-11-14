import os, time, uuid, math, json, argparse, random, pickle
from pathlib import Path
from typing import Iterable, Callable, Dict, List, Tuple

import numpy as np
import torch
torch.set_float32_matmul_precision('high')
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ========================================================================== #
# ───────────────────────────────  Muon  ─────────────────────────────────── #
# ========================================================================== #
@torch.compile
def NewtonSchulz(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """
    Approximate ‖G‖^{-1/2}·G using a polynomial Newton–Schulz iteration.
    Shape‑agnostic: if rows > cols, the transpose trick is used.
    """
    if steps == 0:
        return G / (G.norm() + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if X.size(0) > X.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if G.size(0) > G.size(1) else X


class Muon(optim.Optimizer):
    """
    Muon: SGD with momentum + Newton–Schulz low‑rank update + unit‑norm re‑scaling.
    Designed for weight tensors reshaped as [rows, ...].
    """
    def __init__(self,
                 params: Iterable[Tensor],
                 lr: float = 0.08,
                 momentum: float = 0.7,
                 nesterov: bool = True,
                 steps: int = 5) -> None:
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, steps=steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        if closure is not None:
            with torch.enable_grad():
                closure()
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            nesterov = group["nesterov"]
            steps = group["steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state.setdefault(p, {})
                buf = state.setdefault("momentum_buffer", torch.zeros_like(grad))
                buf.mul_(mom).add_(grad)
                grad_hat = grad.add(buf, alpha=mom) if nesterov else buf

                # Re-scale to fixed norm
                p.data.mul_((len(p.data) ** 0.5) / (p.data.norm() + 1e-12))

                update = NewtonSchulz(
                    grad_hat.reshape(len(grad_hat), -1),
                    steps=steps
                ).view_as(grad_hat)
                p.data.add_(update, alpha=-lr)


def PolarSVD(G: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Exact polar orthogonal factor via SVD.
    Given 2D matrix G (m x n), returns U @ V^T where G = U Σ V^T.
    """
    assert G.ndim == 2, "G must be 2D (use reshape before calling)"
    if G.norm() <= eps:
        return torch.zeros_like(G)
    X = (G.to(torch.float32)) / (G.norm() + eps)
    try:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    except RuntimeError:
        U, S, Vh = torch.linalg.svd(X.cpu(), full_matrices=False)
        U, Vh = U.to(X.device), Vh.to(X.device)
    Q = U @ Vh
    return Q.to(dtype=G.dtype)


class MuonSVD(optim.Optimizer):
    """
    MuonSVD: momentum SGD + exact polar (SVD) orthogonalization + unit‑norm re‑scaling.
    """
    def __init__(self,
                 params: Iterable[Tensor],
                 lr: float = 0.08,
                 momentum: float = 0.7,
                 nesterov: bool = True) -> None:
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None) -> None:
        if closure is not None:
            with torch.enable_grad():
                closure()
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state.setdefault(p, {})
                buf = state.setdefault("momentum_buffer", torch.zeros_like(grad))
                buf.mul_(mom).add_(grad)
                grad_hat = grad.add(buf, alpha=mom) if nesterov else buf

                # Re-scale to fixed norm
                p.data.mul_((len(p.data) ** 0.5) / (p.data.norm() + 1e-12))

                G2 = grad_hat.reshape(len(grad_hat), -1)
                update = PolarSVD(G2).view_as(grad_hat)
                p.data.add_(update, alpha=-lr)


# ───────────────────────────────────────────────────────────────────────── #
# FineWeb-style token dataset (NanoGPT format: train.bin / val.bin)
# ───────────────────────────────────────────────────────────────────────── #

class TokenDataset(Dataset):
    """
    Simple sequence dataset over a 1D token array stored in a .bin file.
    Assumes tokens are stored as uint16 (NanoGPT-style); adjust dtype as needed.
    """
    def __init__(self, bin_path: str | Path, block_size: int):
        super().__init__()
        self.bin_path = Path(bin_path)
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Bin file not found: {self.bin_path}")
        # NOTE: dtype may need adjustment depending on your preprocessing.
        self.data = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        if len(self.data) <= block_size + 1:
            raise ValueError(f"Data in {self.bin_path} too short for block_size={block_size}")

    def __len__(self) -> int:
        # number of possible contiguous sequences
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx: int):
        # slice from memmap, then convert to torch
        x = np.array(self.data[idx:idx + self.block_size], dtype=np.int64)
        y = np.array(self.data[idx + 1:idx + 1 + self.block_size], dtype=np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)


def get_fineweb_loaders(
    root: str = "data/fineweb",
    block_size: int = 256,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val DataLoaders from train.bin / val.bin under root.
    """
    root = Path(root)
    train_bin = root / "train.bin"
    val_bin   = root / "val.bin"
    train_ds = TokenDataset(train_bin, block_size)
    val_ds   = TokenDataset(val_bin, block_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4 if device == "cuda" else 0
    pin_memory = device == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


# ───────────────────────────────────────────────────────────────────────── #
# NanoGPT-style model
# ───────────────────────────────────────────────────────────────────────── #

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
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)  # (B,T,3*C)
        q, k, v = qkv.split(C, dim=2)

        # (B, nh, T, hd)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v  # (B, nh, T, hd)
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

        # optional weight tying
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
        # idx: (B, T), int64
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(f"Sequence length {T} > block_size {self.config.block_size}")
        pos = torch.arange(0, T, device=idx.device, dtype=torch.long).unsqueeze(0)

        tok_emb = self.tok_emb(idx)        # (B, T, C)
        pos_emb = self.pos_emb(pos)        # (1, T, C)
        x = self.drop(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)           # (B, T, vocab_size)
        return logits


def compiled_gpt(config: GPTConfig) -> nn.Module:
    """Return compiled NanoGPT model, falls back gracefully if compile fails."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(config).to(device)
    try:
        return torch.compile(model, mode="max-autotune")
    except Exception:
        return model


#  ───────────────────────────────────────────────────────────────────────── #
# Utilities: evaluation, optimizers, schedulers, run_experiment()
# ───────────────────────────────────────────────────────────────────────── #

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device | str) -> Tuple[float, float]:
    """
    Evaluate average loss and token-level accuracy on a loader.
    Returns (avg_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)  # (B,T,V)
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, V),
                y.view(B * T),
                reduction="sum",
            )
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            total_correct += (preds == y).sum().item()
            total_tokens += y.numel()
    avg_loss = total_loss / max(1, total_tokens)
    acc = total_correct / max(1, total_tokens)
    return avg_loss, acc


def _make_sgd(
    param_groups,
    lr: float,
    momentum: float,
    nesterov: bool = True,
) -> optim.SGD:
    """Create SGD, falling back if fused is unsupported."""
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
    Build Muon / MuonSVD / SGD optimizers for NanoGPT.
    - Muon / MuonSVD act on 'matrix-like' params (ndim >= 2: embeddings, linear weights).
    - Small SGD optimizer handles 1D params (biases, LayerNorm scales, etc.).
    """
    hp = hparams or {}

    matrix_like: list[Tensor] = []
    others: list[Tensor] = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
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

    # Main optimizer (matrix-like weights)
    lr_main = hp.get("lr", {
        "muon":          0.08,
        "muon_svd":      0.08,
        "sgd":           0.08,
    }[name])

    if name == "muon":
        opt_main = Muon(
            [{"params": matrix_like, "lr": lr_main}],
            lr=lr_main,
            momentum=hp.get("momentum", 0.7),
            nesterov=hp.get("nesterov", True),
            steps=hp.get("steps", 5),
        )
    elif name == "muon_svd":
        opt_main = MuonSVD(
            [{"params": matrix_like, "lr": lr_main}],
            lr=lr_main,
            momentum=hp.get("momentum", 0.7),
            nesterov=hp.get("nesterov", True),
        )
    elif name == "sgd":
        opt_main = _make_sgd(
            [{"params": matrix_like, "lr": lr_main}],
            lr=lr_main,
            momentum=hp.get("momentum", 0.7),
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
    """Warm‑up + Cosine LR schedule applied to all optimizers."""
    total_steps = steps_per_epoch * epochs
    warm_steps  = int(total_steps * warmup_ratio)
    def schedule(step: int, w=warm_steps, t=total_steps) -> float:
        warm = 1.0 if w == 0 else min(1.0, step / w)
        cosine = 0.5 * (1 + math.cos(math.pi * max(0, step - w) / max(1, t - w)))
        return warm * cosine
    return [optim.lr_scheduler.LambdaLR(opt, lr_lambda=schedule) for opt in optimizers]


def run_experiment(
    opt_name: str,
    runs: int,
    epochs: int,
    model_fn: Callable[[], nn.Module],
    hparams: dict | None = None,
    batch_size: int = 64,
    data_root: str = "data/fineweb",
    block_size: int = 256,
) -> dict[str, List[List[float]]]:
    """
    Train a single optimizer (multiple runs) on FineWeb + NanoGPT, return:
    {
      "train": [[... per-epoch loss ...] * runs],
      "acc":   [[... per-epoch token-acc ...] * runs],
      "time":  [[... elapsed seconds ...] * runs]
    }
    """
    results = {"train": [], "acc": [], "time": []}
    steps_str = (hparams or {}).get('steps', 'N/A')
    print(f"\n── {opt_name.upper()} (steps={steps_str}) ──")
    print(" | ".join(f"{c:^10}" for c in ("run", "epoch", "train", "val", "sec")))

    for run in range(1, runs + 1):
        torch.manual_seed(run)
        np.random.seed(run)
        random.seed(run)

        model = model_fn()
        device = next(model.parameters()).device

        train_loader, val_loader = get_fineweb_loaders(
            root=data_root,
            block_size=block_size,
            batch_size=batch_size,
        )
        optimizers = build_optimizers(opt_name, model, hparams)
        schedulers = build_schedulers(optimizers, len(train_loader), epochs)

        t0 = time.perf_counter()
        run_loss, run_acc, run_sec = [], [], []

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            nb = 0

            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(x)
                B, T, V = logits.shape
                loss = F.cross_entropy(
                    logits.view(B * T, V),
                    y.view(B * T),
                )
                loss.backward()

                clip_grad_norm_(model.parameters(), 1.0)

                for opt, sch in zip(optimizers, schedulers):
                    opt.step()
                    sch.step()
                model.zero_grad(set_to_none=True)

                epoch_loss += loss.item()
                nb += 1

            epoch_loss /= max(1, nb)
            _, val_acc = evaluate(model, val_loader, device)
            elapsed = time.perf_counter() - t0

            run_loss.append(epoch_loss)
            run_acc.append(val_acc)
            run_sec.append(elapsed)

            print(" | ".join(
                f"{v:^10.4f}" if isinstance(v, float) else f"{v:^10}"
                for v in (run, epoch, run_loss[-1], val_acc, elapsed)
            ))

        results["train"].append(run_loss)
        results["acc"].append(run_acc)
        results["time"].append(run_sec)
        print("-" * 60)

    return results


# ====================================================================== #
# Main: run experiments and save JSON
# ====================================================================== #
def build_gpt_config_from_args(args) -> Tuple[GPTConfig, int, dict]:
    """
    Build GPTConfig, optionally reading vocab_size from meta.pkl if present.
    Returns (config, vocab_size, meta_dict_or_empty).
    """
    root = Path(args.data_root)
    vocab_size = args.vocab_size
    meta_extra: dict = {}

    meta_path = root / "meta.pkl"
    if meta_path.exists():
        try:
            with open(meta_path, "rb") as f:
                meta_extra = pickle.load(f)
            if isinstance(meta_extra, dict) and "vocab_size" in meta_extra:
                vocab_size = int(meta_extra["vocab_size"])
        except Exception:
            meta_extra = {}

    cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    return cfg, vocab_size, meta_extra


def main():
    parser = argparse.ArgumentParser(
        description="Compare Muon / Muon-SVD / SGD-M on FineWeb + NanoGPT."
    )
    # Experiment config
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3],
                        help="Newton–Schulz steps to test for Muon.")
    parser.add_argument("--include_sgd", action="store_true", default=True)
    parser.add_argument("--include_svd", action="store_true", default=True)
    parser.add_argument("--logdir", type=str, default=None,
                        help="Directory to save results. Default: logs/fineweb_nanogpt_cmp_<uuid>")

    # Optimizer base hyperparams
    parser.add_argument("--base_lr", type=float, default=0.08)
    parser.add_argument("--base_momentum", type=float, default=0.7)
    parser.add_argument("--sgd_epochs", type=int, default=6,
                        help="Epochs for SGD with Momentum only (default: 6).")

    # Data / model hyperparams
    parser.add_argument("--data_root", type=str, default="data/fineweb",
                        help="Root directory containing train.bin / val.bin (FineWeb tokenized).")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=50257,
                        help="Default vocab size (overridden by meta.pkl if present).")
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--n_embd", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    # Global perf flags
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42); np.random.seed(42); random.seed(42)

    logdir = Path(args.logdir or (Path("logs") / f"fineweb_nanogpt_cmp_{uuid.uuid4().hex[:8]}"))
    logdir.mkdir(parents=True, exist_ok=True)

    # Build GPT config
    gpt_config, vocab_size, meta_extra = build_gpt_config_from_args(args)

    def make_compiled_gpt() -> nn.Module:
        return compiled_gpt(gpt_config)

    # (Optional) tiny warmup for compilation and kernels
    print("🔥 One-time warm-up on FineWeb + NanoGPT...")
    _ = run_experiment(
        opt_name="muon",
        runs=1,
        epochs=1,
        model_fn=make_compiled_gpt,
        hparams={"steps": 5},
        batch_size=args.batch_size,
        data_root=args.data_root,
        block_size=args.block_size,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Experiment configs: Muon steps, SGD-M, Muon-SVD
    EXPERIMENT_CONFIGS = [
        {
            "name": f"Muon (steps={s})",
            "opt_name": "muon",
            "hparams": {"lr": args.base_lr, "momentum": args.base_momentum, "steps": s},
        }
        for s in args.steps
    ]

    if args.include_sgd:
        EXPERIMENT_CONFIGS.append(
            {
                "name": "SGD with Momentum",
                "opt_name": "sgd",
                "hparams": {"lr": args.base_lr, "momentum": args.base_momentum, "nesterov": True},
            }
        )

    if args.include_svd:
        EXPERIMENT_CONFIGS.append(
            {
                "name": "Muon with SVD",
                "opt_name": "muon_svd",
                "hparams": {"lr": args.base_lr, "momentum": args.base_momentum, "nesterov": True},
            }
        )

    ALL_RESULTS: Dict[str, dict] = {}
    per_experiment_epochs: Dict[str, int] = {}
    per_experiment_runs: Dict[str, int] = {}

    for cfg in EXPERIMENT_CONFIGS:
        runs_this = args.runs
        epochs_this = args.sgd_epochs if cfg["opt_name"] == "sgd" else args.epochs

        per_experiment_runs[cfg["name"]] = runs_this
        per_experiment_epochs[cfg["name"]] = epochs_this

        ALL_RESULTS[cfg["name"]] = run_experiment(
            opt_name = cfg["opt_name"],
            runs     = runs_this,
            epochs   = epochs_this,
            model_fn = make_compiled_gpt,
            hparams  = cfg["hparams"],
            batch_size = args.batch_size,
            data_root = args.data_root,
            block_size = args.block_size,
        )

    # Save results & metadata to JSON
    res_path = logdir / "results.json"
    meta_path = logdir / "metadata.json"
    with open(res_path, "w") as f:
        json.dump(ALL_RESULTS, f)
    with open(meta_path, "w") as f:
        json.dump(
            {
                "dataset": "FineWeb",
                "model": "NanoGPT",
                "runs_default": args.runs,
                "epochs_default": args.epochs,
                "sgd_epochs": args.sgd_epochs,
                "base_lr": args.base_lr,
                "base_momentum": args.base_momentum,
                "steps": args.steps,
                "include_sgd": args.include_sgd,
                "include_svd": args.include_svd,
                "batch_size": args.batch_size,
                "data_root": args.data_root,
                "vocab_size": vocab_size,
                "block_size": args.block_size,
                "n_layer": args.n_layer,
                "n_head": args.n_head,
                "n_embd": args.n_embd,
                "dropout": args.dropout,
                "per_experiment_epochs": per_experiment_epochs,
                "per_experiment_runs": per_experiment_runs,
                "meta_extra": {k: v for k, v in meta_extra.items()
                               if k not in {"vocab_size"}},  # avoid redundancy
            },
            f,
            indent=2,
        )

    print(f"\n✅ Finished. Results saved to:\n  - {res_path}\n  - {meta_path}")
    print("Next: python plot.py --results", str(res_path))


if __name__ == "__main__":
    main()
