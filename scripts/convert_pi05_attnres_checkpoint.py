from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import safetensors.torch
import torch

from openpi.models_pytorch.attnres import PI05AttnResPytorch
from openpi.training import config as _config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert legacy pi0.5 + attnres_state.pt checkpoints into standalone PI05AttnRes checkpoints."
    )
    parser.add_argument("--source-ckpt", required=True, type=Path)
    parser.add_argument("--output-ckpt", required=True, type=Path)
    parser.add_argument("--config", default="pi05_attnres_libero")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_ckpt = args.source_ckpt.expanduser().resolve()
    output_ckpt = args.output_ckpt.expanduser().resolve()
    if not (source_ckpt / "model.safetensors").exists():
        raise FileNotFoundError(f"Missing source model.safetensors: {source_ckpt}")
    if not (source_ckpt / "attnres_state.pt").exists():
        raise FileNotFoundError(f"Missing source attnres_state.pt: {source_ckpt}")
    if output_ckpt.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_ckpt} exists; pass --overwrite to replace it")
        shutil.rmtree(output_ckpt)
    output_ckpt.mkdir(parents=True)

    train_config = _config.get_config(args.config)
    model_config = train_config.model
    model = PI05AttnResPytorch(
        model_config,
        num_blocks=model_config.attnres_num_blocks,
        adapter_rank=model_config.attnres_adapter_rank,
        no_adapter=model_config.attnres_no_adapter,
    ).to(args.device)
    model.load_base_weights(str(source_ckpt / "model.safetensors"), device=args.device)
    model.load_attnres_state(str(source_ckpt / "attnres_state.pt"), device=args.device)
    safetensors.torch.save_model(model, output_ckpt / "model.safetensors")

    assets_src = source_ckpt / "assets"
    if assets_src.exists():
        shutil.copytree(assets_src, output_ckpt / "assets")

    metadata = {}
    metadata_src = source_ckpt / "metadata.pt"
    if metadata_src.exists():
        metadata = torch.load(metadata_src, map_location="cpu", weights_only=False)
    metadata.update(
        {
            "config_name": args.config,
            "standalone_attnres": True,
            "source_checkpoint": str(source_ckpt),
        }
    )
    torch.save(metadata, output_ckpt / "metadata.pt")
    print(f"wrote standalone PI05AttnRes checkpoint: {output_ckpt}")


if __name__ == "__main__":
    main()
