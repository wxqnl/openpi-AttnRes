#!/usr/bin/env python3
"""Prepare the PyTorch-format pi0.5 base checkpoint used by RLinf's SFT pipeline.

This is a single-file, self-contained helper that:

  1. Downloads the official JAX orbax checkpoint of pi0.5_base from the public
     ``gs://openpi-assets/checkpoints/pi05_base`` bucket (anonymous GCS access;
     12.44 GB across 29 files). Resumable - files already on disk at the right
     size are skipped.

  2. Runs openpi's official ``examples/convert_jax_model_to_pytorch.py`` on the
     downloaded checkpoint to produce ``<torch_dir>/model.safetensors`` plus a
     ``config.json`` sidecar.

  3. Copies the ``assets/`` subdirectory (per-robot norm stats) alongside the
     converted weights. (The convert script looks for assets in a sibling path
     and silently skips them when the JAX dir is self-contained.)

Re-running is safe: an already-complete ``--jax-dir`` is detected and download
is skipped; an already-converted ``--torch-dir`` is detected and conversion is
skipped. Use ``--force-download`` / ``--force-convert`` to override.

Precision
---------
Default is ``float32`` because that reproduces RLinf's SFT starting point
**exactly**. openpi's ``PaliGemmaWithExpertModel.__init__`` re-promotes a
whitelist of 122 tensors (RMSNorms, vision patch/position embedding, action
in/out projection, time MLP in/out) back to fp32 during model construction;
loading a fp32 safetensors into such a model preserves full precision on every
tensor. Saving as ``bfloat16`` is ~2x smaller (~7.5 GB) but introduces ~0.4%
rounding error on those 122 whitelisted tensors - negligible for task metrics,
but loss curves will not bit-match the official RLinf baseline.

Requirements
------------
On the machine running this script you need:

* Python 3.10+
* A local checkout of https://github.com/RLinf/openpi (or the Physical
  Intelligence upstream) that contains ``examples/convert_jax_model_to_pytorch.py``.
  Pass the path via ``--openpi-repo`` or the ``OPENPI_REPO`` env var.
* Python deps (all pulled in by openpi's pyproject): ``fsspec``, ``gcsfs``,
  ``torch``, ``safetensors``, ``flax``, ``orbax-checkpoint``, ``jax``, ``tyro``.
  If you have an openpi venv set up, activate it before running this script.
* ~30 GB free disk: ~12 GB JAX checkpoint + ~14.5 GB fp32 PyTorch output
  (or ~7.5 GB with ``--precision bfloat16``).
* Outbound HTTPS to storage.googleapis.com (the bucket is public - no GCP auth
  needed).

Usage
-----
First run (fresh machine)::

    python scripts/prepare_pi05_base_pytorch.py \
        --jax-dir   ./models/pi05_base_jax \
        --torch-dir ./models/pi05_base_pytorch \
        --openpi-repo /path/to/openpi

Re-run in the same project (download + conversion auto-skipped if complete)::

    python scripts/prepare_pi05_base_pytorch.py \
        --jax-dir   ./models/pi05_base_jax \
        --torch-dir ./models/pi05_base_pytorch \
        --openpi-repo /path/to/openpi

Smaller bf16 output (not bit-identical to RLinf SFT starting point)::

    python scripts/prepare_pi05_base_pytorch.py ... --precision bfloat16
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import pathlib
import shutil
import subprocess
import sys
import time

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

GCS_BUCKET = "openpi-assets"
GCS_PREFIX = "checkpoints/pi05_base"

# Any openpi training config that constructs ``Pi0Config(pi05=True)`` works -
# the convert script only reads model architecture fields (paligemma_variant,
# action_expert_variant, pi05, action_dim, ...), not the data / loss config.
DEFAULT_CONFIG_NAME = "pi05_aloha"


# ----------------------------------------------------------------------------
# Step 1: download JAX orbax checkpoint from GCS
# ----------------------------------------------------------------------------


def parallel_download_jax(dest: pathlib.Path, workers: int) -> None:
    """Mirror gs://openpi-assets/checkpoints/pi05_base into ``dest`` (resumable)."""
    try:
        import fsspec  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "fsspec/gcsfs is required for the download step. Install with: pip install 'fsspec[gcs]'"
        ) from e

    remote_root = f"{GCS_BUCKET}/{GCS_PREFIX}"
    fs = fsspec.filesystem("gcs", token="anon")

    entries = fs.find(remote_root, detail=True, withdirs=False)
    files = [(name, meta) for name, meta in entries.items() if meta["type"] == "file"]
    total_bytes = sum((m.get("size") or 0) for _, m in files)
    print(
        f"[download] remote gs://{remote_root} -> {dest}\n"
        f"[download] {len(files)} files, {total_bytes / 1e9:.2f} GB, workers={workers}"
    )
    dest.mkdir(parents=True, exist_ok=True)

    def rel_local(remote_name: str) -> pathlib.Path:
        assert remote_name.startswith(remote_root + "/"), remote_name
        return dest / remote_name[len(remote_root) + 1 :]

    def fetch(item):
        name, meta = item
        size = int(meta.get("size") or 0)
        local = rel_local(name)
        if local.exists() and local.stat().st_size == size:
            return "skip", name, size
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_suffix(local.suffix + ".part")
        with fs.open("gs://" + name, "rb") as r, open(tmp, "wb") as w:
            while True:
                buf = r.read(8 * 1024 * 1024)  # 8 MiB chunks
                if not buf:
                    break
                w.write(buf)
        tmp.rename(local)
        return "done", name, size

    t0 = time.time()
    done_bytes = 0
    n_done = 0
    n_skip = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for status, name, size in pool.map(fetch, files):
            if status == "skip":
                n_skip += 1
            else:
                n_done += 1
            done_bytes += size
            elapsed = time.time() - t0
            rate = (done_bytes / 1e6) / max(elapsed, 1e-3)
            print(
                f"[download] [{n_done + n_skip:3d}/{len(files)}] {status:4s}  "
                f"{size / 1e6:7.1f} MB  total {done_bytes / 1e9:5.2f} GB  "
                f"elapsed {elapsed:6.1f}s  rate {rate:5.1f} MB/s  {name}",
                flush=True,
            )

    elapsed = time.time() - t0
    print(f"[download] finished: {n_done} downloaded, {n_skip} skipped, {done_bytes / 1e9:.2f} GB in {elapsed:.1f}s.")


def is_jax_ckpt_ready(jax_dir: pathlib.Path) -> bool:
    """Cheap sanity check that a local JAX orbax ckpt is complete."""
    if not jax_dir.is_dir():
        return False
    sentinels = [
        jax_dir / "params" / "_METADATA",
        jax_dir / "params" / "_sharding",
        jax_dir / "params" / "manifest.ocdbt",
        jax_dir / "params" / "commit_success.txt",
    ]
    if not all(p.is_file() for p in sentinels):
        return False
    total = sum(p.stat().st_size for p in jax_dir.rglob("*") if p.is_file())
    # The full checkpoint is ~12.44 GB; 10 GB is a safe lower bound.
    return total > 10 * 1024**3


# ----------------------------------------------------------------------------
# Step 2: JAX -> PyTorch conversion (shells out to openpi's official script)
# ----------------------------------------------------------------------------


def run_convert(
    openpi_repo: pathlib.Path,
    jax_dir: pathlib.Path,
    torch_dir: pathlib.Path,
    precision: str,
    config_name: str,
) -> None:
    script = openpi_repo / "examples" / "convert_jax_model_to_pytorch.py"
    if not script.is_file():
        raise FileNotFoundError(
            f"Convert script not found at {script}. "
            f"Pass --openpi-repo pointing at a checkout of "
            f"https://github.com/RLinf/openpi.git (or Physical Intelligence's upstream)."
        )
    cmd = [
        sys.executable,
        "-u",
        str(script),
        "--checkpoint_dir",
        str(jax_dir),  # parent dir of params/ and assets/
        "--config_name",
        config_name,
        "--output_path",
        str(torch_dir),
        "--precision",
        precision,
    ]
    env = os.environ.copy()
    # This is a pure weight-conversion job; no need to initialize GPU.
    env.setdefault("JAX_PLATFORMS", "cpu")
    print(f"[convert] cwd={openpi_repo}\n[convert] cmd: {' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=str(openpi_repo), env=env, check=False).returncode
    if rc != 0:
        raise RuntimeError(f"convert_jax_model_to_pytorch.py failed with exit code {rc}")


def is_torch_ckpt_ready(torch_dir: pathlib.Path) -> bool:
    return (torch_dir / "model.safetensors").is_file() and (torch_dir / "config.json").is_file()


# ----------------------------------------------------------------------------
# Step 3: copy assets/ (per-robot norm_stats.json) next to the converted model
# ----------------------------------------------------------------------------


def copy_assets(jax_dir: pathlib.Path, torch_dir: pathlib.Path) -> None:
    src = jax_dir / "assets"
    if not src.is_dir():
        print(f"[assets] source missing at {src}, skip.")
        return
    dst = torch_dir / "assets"
    if dst.is_dir():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    n = sum(1 for _ in dst.rglob("*") if _.is_file())
    print(f"[assets] copied {src} -> {dst}  ({n} files)")


# ----------------------------------------------------------------------------
# CLI entrypoint
# ----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download the pi0.5 base JAX checkpoint from gs://openpi-assets and "
            "convert it to PyTorch for use as an RLinf SFT starting point."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--jax-dir",
        type=pathlib.Path,
        required=True,
        help="Local directory to mirror gs://openpi-assets/checkpoints/pi05_base into "
        "(will contain params/ and assets/).",
    )
    parser.add_argument(
        "--torch-dir",
        type=pathlib.Path,
        required=True,
        help="Output directory for the converted PyTorch model "
        "(contains model.safetensors, config.json, and a copy of assets/).",
    )

    default_openpi_env = os.environ.get("OPENPI_REPO")
    parser.add_argument(
        "--openpi-repo",
        type=pathlib.Path,
        default=pathlib.Path(default_openpi_env) if default_openpi_env else None,
        help="Path to an openpi source checkout that contains "
        "examples/convert_jax_model_to_pytorch.py. "
        "Falls back to the OPENPI_REPO env var.",
    )
    parser.add_argument(
        "--precision",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        help=(
            "Saved dtype for ALL tensors in the PyTorch safetensors. 'float32' "
            "matches RLinf's SFT starting point exactly (the whitelist in "
            "PaliGemmaWithExpertModel keeps norms/embeddings/action_proj/time_mlp "
            "in fp32 during training). 'bfloat16' is ~7.5 GB vs ~14.5 GB but "
            "rounds those 122 whitelisted tensors with ~0.4%% relative error."
        ),
    )
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="openpi training config name, used only for model architecture. "
        "Any config that uses Pi0Config(pi05=True) at default variants works.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=16,
        help="Parallel connections for the GCS download.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload the JAX checkpoint even if --jax-dir already looks complete.",
    )
    parser.add_argument(
        "--force-convert",
        action="store_true",
        help="Rerun the conversion even if --torch-dir/model.safetensors exists.",
    )
    args = parser.parse_args()

    if args.openpi_repo is None:
        parser.error("--openpi-repo is required (or set the OPENPI_REPO env var).")
    args.openpi_repo = args.openpi_repo.resolve()
    args.jax_dir = args.jax_dir.resolve()
    args.torch_dir = args.torch_dir.resolve()

    # Step 1: download
    if args.force_download or not is_jax_ckpt_ready(args.jax_dir):
        parallel_download_jax(args.jax_dir, workers=args.download_workers)
    else:
        print(f"[download] skip: {args.jax_dir} already has a complete JAX checkpoint.")

    # Step 2: convert
    if args.force_convert or not is_torch_ckpt_ready(args.torch_dir):
        run_convert(
            openpi_repo=args.openpi_repo,
            jax_dir=args.jax_dir,
            torch_dir=args.torch_dir,
            precision=args.precision,
            config_name=args.config_name,
        )
    else:
        print(f"[convert] skip: {args.torch_dir / 'model.safetensors'} already exists. Use --force-convert to redo.")

    # Step 3: always ensure assets/ is present alongside the model
    copy_assets(args.jax_dir, args.torch_dir)

    print(f"[done] PyTorch pi05_base ready at: {args.torch_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
