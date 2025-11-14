import os, time, uuid, math, json, argparse, random
from pathlib import Path
from typing import Iterable, Callable, Dict, List, Tuple

import numpy as np
import torch
import torchvision
torch.set_float32_matmul_precision('high')
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image

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
# Tiny-ImageNet dataloading & WideResNet-28-10 backbone
# ───────────────────────────────────────────────────────────────────────── #

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class TinyImageNetDataset(Dataset):
    """
    Tiny-ImageNet-200 dataset loader for the original directory structure:
      root/
        wnids.txt
        train/<wnid>/images/*.JPEG
        val/images/*.JPEG
        val/val_annotations.txt
    """
    def __init__(self, root: str | Path, split: str = "train", transform=None):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform

        wnids_path = self.root / "wnids.txt"
        if not wnids_path.exists():
            raise FileNotFoundError(f"wnids.txt not found under {self.root}. "
                                    f"Expected Tiny-ImageNet-200 layout.")

        with open(wnids_path, "r") as f:
            self.wnids = [line.strip() for line in f if line.strip()]
        self.class_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}

        self.samples: list[Tuple[Path, int]] = []

        if split == "train":
            train_dir = self.root / "train"
            for wnid in self.wnids:
                img_dir = train_dir / wnid / "images"
                if not img_dir.exists():
                    continue
                for img_file in img_dir.glob("*.JPEG"):
                    self.samples.append((img_file, self.class_to_idx[wnid]))
        elif split in ("val", "valid", "validation"):
            val_dir = self.root / "val"
            ann_path = val_dir / "val_annotations.txt"
            img_dir = val_dir / "images"
            if not ann_path.exists():
                raise FileNotFoundError(f"val_annotations.txt not found under {val_dir}")
            if not img_dir.exists():
                raise FileNotFoundError(f"val/images/ not found under {val_dir}")

            with open(ann_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    fname, wnid = parts[0], parts[1]
                    if wnid not in self.class_to_idx:
                        continue
                    img_path = img_dir / fname
                    self.samples.append((img_path, self.class_to_idx[wnid]))
        else:
            raise ValueError(f"Unknown split '{split}'. Use 'train' or 'val'.")

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found for split='{split}' under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def get_tinyimagenet_loaders(
    root: str = "tiny-imagenet-200",
    batch_size: int = 128,
) -> Tuple[DataLoader, DataLoader]:
    """Standard Tiny-ImageNet loaders (64x64) with crop + flip augmentations."""
    transform_train = T.Compose([
        T.RandomCrop(64, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    transform_val = T.Compose([
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_set = TinyImageNetDataset(root=root, split="train", transform=transform_train)
    val_set   = TinyImageNetDataset(root=root, split="val",   transform=transform_val)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4 if device == "cuda" else 0
    pin_memory = device == "cuda"

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


# ---------------- WideResNet-28-10 (CIFAR-style) ------------------------

class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

        self.equal_in_out = (in_planes == out_planes)
        self.conv_shortcut = None
        if not self.equal_in_out:
            self.conv_shortcut = nn.Conv2d(
                in_planes, out_planes, kernel_size=1, stride=stride, bias=False
            )

    def forward(self, x):
        if not self.equal_in_out:
            out = self.relu1(self.bn1(x))
            shortcut = self.conv_shortcut(out)
        else:
            out = self.relu1(self.bn1(x))
            shortcut = x
        out = self.conv1(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = out + shortcut
        return out


class WideResNetBlock(nn.Module):
    def __init__(self, num_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                block(
                    in_planes if i == 0 else out_planes,
                    out_planes,
                    stride if i == 0 else 1,
                    drop_rate=drop_rate,
                )
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class WideResNet(nn.Module):
    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 10,
        num_classes: int = 200,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WideResNet depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor

        n_channels = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = WideResNetBlock(n, n_channels[0], n_channels[1], WideBasicBlock, stride=1, drop_rate=drop_rate)
        self.block2 = WideResNetBlock(n, n_channels[1], n_channels[2], WideBasicBlock, stride=2, drop_rate=drop_rate)
        self.block3 = WideResNetBlock(n, n_channels[2], n_channels[3], WideBasicBlock, stride=2, drop_rate=drop_rate)
        self.bn = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(n_channels[3], num_classes)
        self.n_channels = n_channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), self.n_channels)
        out = self.fc(out)
        return out


def make_wrn28_10(num_classes: int = 200) -> nn.Module:
    return WideResNet(depth=28, widen_factor=10, num_classes=num_classes, drop_rate=0.0)


def compiled_wrn28_10() -> nn.Module:
    """Return compiled WideResNet-28-10 model (channels_last), falls back gracefully."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_wrn28_10(num_classes=200).to(device).to(memory_format=torch.channels_last)
    try:
        return torch.compile(model, mode="max-autotune")
    except Exception:
        return model


#  ───────────────────────────────────────────────────────────────────────── #
# Utilities: evaluate(), optimizer/scheduler builders, run_experiment()
# ───────────────────────────────────────────────────────────────────────── #

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device | str) -> float:
    """Accuracy over the whole loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / max(1, total)


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
    hp = hparams or {}

    # Parameter groups
    conv_filt = [p for p in model.parameters() if p.ndim == 4 and p.requires_grad]

    norm: list[Tensor] = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            for p in m.parameters():
                if p.requires_grad:
                    norm.append(p)

    head: list[Tensor] = []
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        for p in model.fc.parameters():
            if p.requires_grad:
                head.append(p)

    # Aux (small) optimizer for non‑conv filters / head
    lr_other  = hp.get('lr_other', 0.001)
    lr_head   = hp.get('lr_head',  0.1)
    mom_other = hp.get('momentum_other', 0.99)

    small_groups = []
    if norm:
        small_groups.append(
            {"params": norm, "lr": lr_other, "momentum": mom_other, "nesterov": True}
        )
    if head:
        small_groups.append(
            {"params": head, "lr": lr_head, "momentum": mom_other, "nesterov": True}
        )

    opt_small = _make_sgd(
        small_groups, lr=lr_other, momentum=mom_other, nesterov=True
    ) if small_groups else None

    # Main optimizer (conv filters)
    lr_main = hp.get("lr", {
        "muon":          0.08,
        "muon_svd":      0.08,
        "sgd":           0.08,
    }[name])

    if name == "muon":
        opt_main = Muon(
            [{"params": conv_filt, "lr": lr_main}],
            lr=lr_main,
            momentum=hp.get("momentum", 0.7),
            nesterov=hp.get("nesterov", True),
            steps=hp.get("steps", 5)
        )
    elif name == "muon_svd":
        opt_main = MuonSVD(
            [{"params": conv_filt, "lr": lr_main}],
            lr=lr_main,
            momentum=hp.get("momentum", 0.7),
            nesterov=hp.get("nesterov", True),
        )
    elif name == "sgd":
        opt_main = _make_sgd(
            [{"params": conv_filt, "lr": lr_main}],
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
    batch_size: int = 128,
    data_root: str = "tiny-imagenet-200",
) -> dict[str, List[List[float]]]:
    """
    Train a single optimizer (multiple runs) on Tiny-ImageNet + WideResNet-28-10, return:
    {
      "train": [[... per‑epoch loss ...] * runs],
      "acc":   [[... per‑epoch acc  ...] * runs],
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

        train_loader, valid_loader = get_tinyimagenet_loaders(
            root=data_root,
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
                loss = F.cross_entropy(logits, y, label_smoothing=0.2)
                loss.backward()

                clip_grad_norm_(model.parameters(), 1.0)

                for opt, sch in zip(optimizers, schedulers):
                    opt.step()
                    sch.step()
                model.zero_grad(set_to_none=True)

                epoch_loss += loss.item()
                nb += 1

            epoch_loss /= max(1, nb)
            val_acc = evaluate(model, valid_loader, device)
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
def main():
    parser = argparse.ArgumentParser(
        description="Compare Muon / Muon-SVD / SGD-M on Tiny-ImageNet + WideResNet-28-10."
    )
    # Defaults: 5 runs, 50 epochs for all; SGDM override is 55
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3],
                        help="Newton–Schulz steps to test for Muon.")
    parser.add_argument("--include_sgd", action="store_true", default=True)
    parser.add_argument("--include_svd", action="store_true", default=True)
    parser.add_argument("--logdir", type=str, default=None,
                        help="Directory to save results. Default: logs/tinyimagenet_wrn28_10_cmp_<uuid>")
    parser.add_argument("--base_lr", type=float, default=0.08)
    parser.add_argument("--base_momentum", type=float, default=0.7)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_root", type=str, default="tiny-imagenet-200",
                        help="Root of Tiny-ImageNet-200 (contains wnids.txt, train/, val/).")
    # SGDM-specific epoch override (defaults to 55)
    parser.add_argument("--sgd_epochs", type=int, default=55,
                        help="Epochs for SGD with Momentum only (default: 55).")
    args = parser.parse_args()

    # Global perf flags
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42); np.random.seed(42); random.seed(42)

    logdir = Path(args.logdir or (Path("logs") / f"tinyimagenet_wrn28_10_cmp_{uuid.uuid4().hex[:8]}"))
    logdir.mkdir(parents=True, exist_ok=True)

    # (Optional) tiny warmup for compilation and kernels
    print("🔥 One-time warm-up on Tiny-ImageNet...")
    _ = run_experiment(
        opt_name="muon",
        runs=1,
        epochs=1,
        model_fn=compiled_wrn28_10,
        hparams={"steps": 5},
        batch_size=args.batch_size,
        data_root=args.data_root,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

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
        # 5 runs for all; 50 epochs for others, 55 for SGDM by default
        runs_this = args.runs
        epochs_this = args.sgd_epochs if cfg["opt_name"] == "sgd" else args.epochs

        per_experiment_runs[cfg["name"]] = runs_this
        per_experiment_epochs[cfg["name"]] = epochs_this

        ALL_RESULTS[cfg["name"]] = run_experiment(
            opt_name = cfg["opt_name"],
            runs     = runs_this,
            epochs   = epochs_this,
            model_fn = compiled_wrn28_10,
            hparams  = cfg["hparams"],
            batch_size = args.batch_size,
            data_root = args.data_root,
        )

    # Save results & metadata to JSON
    res_path = logdir / "results.json"
    meta_path = logdir / "metadata.json"
    with open(res_path, "w") as f:
        json.dump(ALL_RESULTS, f)
    with open(meta_path, "w") as f:
        json.dump(
            {
                "dataset": "Tiny-ImageNet-200",
                "model": "WideResNet-28-10 (CIFAR-style)",
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
                "per_experiment_epochs": per_experiment_epochs,
                "per_experiment_runs": per_experiment_runs,
            },
            f,
            indent=2,
        )

    print(f"\n✅ Finished. Results saved to:\n  - {res_path}\n  - {meta_path}")
    print("Next: python plot_results.py --results", str(res_path))


if __name__ == "__main__":
    main()
