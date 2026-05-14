from __future__ import annotations

import argparse
import logging
import socket
import sys
from pathlib import Path

import torch

from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


def install_retrofit(policy, retrofit_dir: Path, state_path: Path, adapter_rank: int, num_blocks: int):
    model = policy._model  # noqa: SLF001 - this is an experiment server.
    if hasattr(model, "attnres"):
        logging.info("Policy model already owns registered PI05 AttnRes; no sidecar retrofit install needed.")
        return model.attnres

    if retrofit_dir is not None:
        sys.path.insert(0, str(retrofit_dir))
        from pi05_paligemma_attnres_retrofit import PI05PaliGemmaAttnResRetrofit
    else:
        from openpi.models_pytorch.attnres import PI05PaliGemmaAttnResRetrofit

    if not hasattr(model, "paligemma_with_expert"):
        raise TypeError("AttnRes LIBERO server requires a PyTorch PI0Pytorch policy model.")

    ckpt = torch.load(state_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    wrapper = PI05PaliGemmaAttnResRetrofit(
        model.paligemma_with_expert.paligemma,
        num_blocks=int(cfg.get("num_blocks", num_blocks)),
        adapter_rank=int(cfg.get("adapter_rank", adapter_rank)),
        no_adapter=bool(cfg.get("no_adapter", False)),
    ).to(policy._pytorch_device)  # noqa: SLF001
    dtype = next(wrapper.parameters()).dtype
    wrapper.router.load_state_dict(
        {k: v.to(device=policy._pytorch_device, dtype=dtype) for k, v in ckpt["router"].items()}  # noqa: SLF001
    )
    if not bool(cfg.get("no_adapter", False)):
        wrapper.adapters.load_state_dict(
            {k: v.to(device=policy._pytorch_device, dtype=dtype) for k, v in ckpt["adapters"].items()}  # noqa: SLF001
        )
    wrapper.gamma.data.copy_(ckpt["gamma"].to(device=policy._pytorch_device, dtype=dtype))  # noqa: SLF001
    wrapper.eval()
    for param in wrapper.parameters():
        param.requires_grad = False

    # Rebind after patching so Policy calls the current method.
    policy._sample_actions = model.sample_actions  # noqa: SLF001
    logging.info("Loaded pi0.5 PaliGemma AttnRes: %s", state_path)
    logging.info("AttnRes gamma mean: %.6f", float(wrapper.gamma.mean().detach().cpu()))
    return wrapper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--checkpoint-dir", default="/home/user01/Minko/openpi/models/pi05_libero_pytorch")
    parser.add_argument(
        "--retrofit-state-path",
        default=None,
        help="AttnRes state path. Defaults to <checkpoint-dir>/attnres_state.pt if present.",
    )
    parser.add_argument("--retrofit-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--adapter-rank", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=9)
    parser.add_argument("--disable-torch-compile", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    retrofit_state_path = (
        Path(args.retrofit_state_path)
        if args.retrofit_state_path is not None
        else checkpoint_dir / "attnres_state.pt"
    )
    if not retrofit_state_path.exists() and args.config != "pi05_attnres_libero":
        raise FileNotFoundError(
            f"AttnRes state not found: {retrofit_state_path}. "
            "Pass --retrofit-state-path explicitly for checkpoints that do not include attnres_state.pt."
        )

    original_compile = getattr(torch, "compile", None)
    if args.disable_torch_compile and original_compile is not None:
        torch.compile = lambda fn, *unused_args, **unused_kwargs: fn
    try:
        policy = _policy_config.create_trained_policy(
            _config.get_config(args.config),
            checkpoint_dir,
            pytorch_device=args.device,
        )
    finally:
        if args.disable_torch_compile and original_compile is not None:
            torch.compile = original_compile
    wrapper = install_retrofit(
        policy,
        Path(args.retrofit_dir) if args.retrofit_dir is not None else None,
        retrofit_state_path,
        args.adapter_rank,
        args.num_blocks,
    )
    if policy.metadata is not None:
        policy.metadata["attnres_state_path"] = str(retrofit_state_path)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating AttnRes LIBERO server (host: %s, ip: %s, port: %d)", hostname, local_ip, args.port)

    # Keep wrapper alive for the process lifetime.
    policy._attnres_wrapper = wrapper  # noqa: SLF001
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
