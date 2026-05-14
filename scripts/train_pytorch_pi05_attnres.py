"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import dataclasses
import gc
import logging
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullOptimStateDictConfig
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import StateDictType
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models.pi05_attnres_config
import openpi.models_pytorch.attnres as attnres_pytorch
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value is not None else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def is_fsdp_model(model) -> bool:
    return isinstance(model, FSDP)


def unwrap_model(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel) or is_fsdp_model(model):
        return model.module
    return model


def _config_value(model_config, field_name: str, default):
    return getattr(model_config, field_name, default)


def install_attnres_retrofit(model, device: torch.device, model_config=None):
    """Mount an AttnRes retrofit on PI0.5's PaliGemma branch."""
    state_path = _env_path("OPENPI_ATTNRES_STATE_PATH")
    retrofit_dir = _env_path("OPENPI_ATTNRES_RETROFIT_DIR")
    default_init = _config_value(model_config, "attnres_init", "pretrained" if state_path is not None else "random")
    init_mode = os.environ.get("OPENPI_ATTNRES_INIT", default_init).lower()
    trainable = _env_bool("OPENPI_ATTNRES_TRAINABLE", _config_value(model_config, "attnres_trainable", False))
    gamma_schedule = _env_bool(
        "OPENPI_ATTNRES_GAMMA_SCHEDULE", _config_value(model_config, "attnres_gamma_schedule", trainable)
    )
    gamma_start = _env_float("OPENPI_ATTNRES_GAMMA_START", _config_value(model_config, "attnres_gamma_start", 0.0))
    gamma_end = _env_float("OPENPI_ATTNRES_GAMMA_END", _config_value(model_config, "attnres_gamma_end", 1.0))
    gamma_ramp_frac = _env_float(
        "OPENPI_ATTNRES_GAMMA_RAMP_FRAC", _config_value(model_config, "attnres_gamma_ramp_frac", 1.0)
    )
    gamma_ramp_steps_env = os.environ.get("OPENPI_ATTNRES_GAMMA_RAMP_STEPS")
    gamma_ramp_steps = (
        int(gamma_ramp_steps_env)
        if gamma_ramp_steps_env is not None
        else _config_value(model_config, "attnres_gamma_ramp_steps", None)
    )

    if init_mode in {"none", "off", "disabled"}:
        return None
    if init_mode not in {"pretrained", "random"}:
        raise ValueError(f"Unsupported OPENPI_ATTNRES_INIT={init_mode!r}; expected pretrained or random")
    if init_mode == "pretrained":
        if state_path is None:
            raise ValueError("OPENPI_ATTNRES_STATE_PATH must be set when OPENPI_ATTNRES_INIT=pretrained")
        if not state_path.exists():
            raise FileNotFoundError(f"AttnRes state not found: {state_path}")
    if not hasattr(model, "paligemma_with_expert"):
        raise TypeError("AttnRes retrofit requires a PI0Pytorch model with paligemma_with_expert")

    ckpt = torch.load(state_path, map_location="cpu", weights_only=False) if init_mode == "pretrained" else {}
    cfg = ckpt.get("config", {})
    num_blocks = int(cfg.get("num_blocks", _env_int("OPENPI_ATTNRES_NUM_BLOCKS", _config_value(model_config, "attnres_num_blocks", 9))))
    adapter_rank = int(
        cfg.get("adapter_rank", _env_int("OPENPI_ATTNRES_ADAPTER_RANK", _config_value(model_config, "attnres_adapter_rank", 256)))
    )
    no_adapter = bool(cfg.get("no_adapter", _env_bool("OPENPI_ATTNRES_NO_ADAPTER", _config_value(model_config, "attnres_no_adapter", False))))

    registered_wrapper = getattr(model, "attnres", None)
    if registered_wrapper is not None:
        wrapper = registered_wrapper
        logging.info("Using registered PI05AttnResPytorch AttnRes module")
    else:
        if retrofit_dir is not None:
            if not retrofit_dir.exists():
                raise FileNotFoundError(f"AttnRes retrofit dir not found: {retrofit_dir}")
            sys.path.insert(0, str(retrofit_dir))
            from pi05_paligemma_attnres_retrofit import PI05PaliGemmaAttnResRetrofit
        else:
            PI05PaliGemmaAttnResRetrofit = attnres_pytorch.PI05PaliGemmaAttnResRetrofit
        wrapper = PI05PaliGemmaAttnResRetrofit(
            model.paligemma_with_expert.paligemma,
            num_blocks=num_blocks,
            adapter_rank=adapter_rank,
            no_adapter=no_adapter,
        ).to(device)
    dtype = next(wrapper.parameters()).dtype
    if init_mode == "pretrained":
        wrapper.router.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in ckpt["router"].items()})
        if not no_adapter:
            wrapper.adapters.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in ckpt["adapters"].items()})
        wrapper.gamma.data.copy_(ckpt["gamma"].to(device=device, dtype=dtype))
    if gamma_schedule:
        wrapper.gamma.data.fill_(gamma_start)

    wrapper.train(trainable)
    for param in wrapper.retrofit_parameters():
        param.requires_grad = trainable

    if registered_wrapper is None:
        # Legacy path for plain PI0Pytorch. Standalone PI05AttnResPytorch
        # registers attnres as a real submodule so it is saved in model.safetensors.
        object.__setattr__(model, "_attnres_wrapper", wrapper)
    object.__setattr__(model, "_attnres_state_path", str(state_path) if state_path is not None else None)
    object.__setattr__(model, "_attnres_init_mode", init_mode)
    object.__setattr__(model, "_attnres_trainable", trainable)
    object.__setattr__(model, "_attnres_adapter_rank", adapter_rank)
    object.__setattr__(model, "_attnres_gamma_schedule", gamma_schedule)
    object.__setattr__(model, "_attnres_gamma_start", gamma_start)
    object.__setattr__(model, "_attnres_gamma_end", gamma_end)
    object.__setattr__(model, "_attnres_gamma_ramp_frac", gamma_ramp_frac)
    object.__setattr__(model, "_attnres_gamma_ramp_steps", gamma_ramp_steps)
    logging.info(
        "Mounted %s pi0.5 PaliGemma AttnRes (trainable=%s, num_blocks=%d, adapter_rank=%d, gamma_schedule=%s)",
        init_mode,
        trainable,
        num_blocks,
        adapter_rank,
        gamma_schedule,
    )
    logging.info("AttnRes gamma mean: %.6f", float(wrapper.gamma.mean().detach().cpu()))
    return wrapper


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """Get state dict from model, handling DDP/FSDP wrappers."""
    return unwrap_model(model).state_dict() if not is_fsdp_model(model) else model.state_dict()


def get_model_parameters(model):
    """Get parameters from model, handling DDP/FSDP wrappers."""
    return model.parameters()


def get_attnres_wrapper(model):
    unwrapped = unwrap_model(model)
    return getattr(unwrapped, "attnres", None) or getattr(unwrapped, "_attnres_wrapper", None)


def attnres_is_registered(model) -> bool:
    return hasattr(unwrap_model(model), "attnres")


def get_attnres_parameters(model, *, trainable_only: bool = False):
    wrapper = get_attnres_wrapper(model)
    if wrapper is None:
        return []
    params = list(wrapper.retrofit_parameters())
    if trainable_only:
        params = [param for param in params if param.requires_grad]
    return params


def get_trainable_parameters(model):
    params = [param for param in get_model_parameters(model) if param.requires_grad]
    if not attnres_is_registered(model):
        params += get_attnres_parameters(model, trainable_only=True)
    return params


def sync_attnres_gradients(model):
    """Average gradients for AttnRes params that live outside the FSDP-wrapped module."""
    if attnres_is_registered(model):
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    for param in get_attnres_parameters(model, trainable_only=True):
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)


def apply_attnres_gamma_schedule(model, step: int, num_train_steps: int) -> float | None:
    wrapper = get_attnres_wrapper(model)
    if wrapper is None or not getattr(unwrap_model(model), "_attnres_gamma_schedule", False):
        return None
    ramp_steps = getattr(unwrap_model(model), "_attnres_gamma_ramp_steps", None)
    if ramp_steps is None:
        ramp_frac = getattr(unwrap_model(model), "_attnres_gamma_ramp_frac", 1.0)
        ramp_steps = max(int(num_train_steps * ramp_frac), 1)
    frac = min(max(step / max(ramp_steps, 1), 0.0), 1.0)
    start = getattr(unwrap_model(model), "_attnres_gamma_start", 0.0)
    end = getattr(unwrap_model(model), "_attnres_gamma_end", 1.0)
    gamma_value = start + (end - start) * frac
    with torch.no_grad():
        wrapper.gamma.data.fill_(gamma_value)
    return float(gamma_value)


def export_attnres_state(model, path: Path) -> bool:
    wrapper = get_attnres_wrapper(model)
    if wrapper is None:
        return False
    config = {
        "num_blocks": int(wrapper.num_blocks),
        "adapter_rank": int(getattr(unwrap_model(model), "_attnres_adapter_rank", 256)),
        "no_adapter": bool(getattr(wrapper, "no_adapter", False)),
        "init": getattr(unwrap_model(model), "_attnres_init_mode", None),
        "trainable": bool(getattr(unwrap_model(model), "_attnres_trainable", False)),
        "gamma_schedule": bool(getattr(unwrap_model(model), "_attnres_gamma_schedule", False)),
        "gamma_start": float(getattr(unwrap_model(model), "_attnres_gamma_start", 0.0)),
        "gamma_end": float(getattr(unwrap_model(model), "_attnres_gamma_end", 1.0)),
        "gamma_ramp_frac": float(getattr(unwrap_model(model), "_attnres_gamma_ramp_frac", 1.0)),
        "gamma_ramp_steps": getattr(unwrap_model(model), "_attnres_gamma_ramp_steps", None),
    }
    state = {
        "config": config,
        "router": {k: v.detach().cpu() for k, v in wrapper.router.state_dict().items()},
        "adapters": {k: v.detach().cpu() for k, v in wrapper.adapters.state_dict().items()},
        "gamma": wrapper.gamma.detach().cpu(),
    }
    torch.save(state, path)
    return True


def save_model_safetensors(model, path: Path, is_main: bool) -> None:
    if is_fsdp_model(model):
        state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_config):
            state_dict = model.state_dict()
        if is_main:
            safetensors.torch.save_file(state_dict, path)
        del state_dict
        return

    if is_main:
        safetensors.torch.save_model(unwrap_model(model), path)


def save_optimizer_state(model, optimizer, path: Path, is_main: bool) -> bool:
    if is_fsdp_model(model):
        if os.environ.get("OPENPI_SAVE_FSDP_OPTIMIZER", "0") != "1":
            if is_main:
                logging.info(
                    "Skipping full FSDP optimizer state save. Set OPENPI_SAVE_FSDP_OPTIMIZER=1 to gather it."
                )
            return False
        optim_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, optim_state_dict_config=optim_config):
            optim_state = FSDP.full_optim_state_dict(model, optimizer, rank0_only=True)
        if is_main:
            torch.save(optim_state, path)
        del optim_state
        return True

    if is_main:
        torch.save(optimizer.state_dict(), path)
    return True


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    # Only save if it's time to save or if it's the final step
    if global_step > 0 and (global_step % config.save_interval == 0 or global_step >= config.num_train_steps):
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        if is_main:
            # Remove any existing temp directory and create new one
            if tmp_ckpt_dir.exists():
                shutil.rmtree(tmp_ckpt_dir)
            tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        if dist.is_initialized():
            dist.barrier()

        save_model_safetensors(model, tmp_ckpt_dir / "model.safetensors", is_main)
        optimizer_state_saved = save_optimizer_state(model, optimizer, tmp_ckpt_dir / "optimizer.pt", is_main)

        model_to_save = unwrap_model(model)
        if is_main:
            # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
            metadata = {
                "global_step": global_step,
                "config": dataclasses.asdict(config),
                "timestamp": time.time(),
                "attnres_state_path": getattr(model_to_save, "_attnres_state_path", None),
                "attnres_init_mode": getattr(model_to_save, "_attnres_init_mode", None),
                "attnres_trainable": getattr(model_to_save, "_attnres_trainable", False),
                "attnres_gamma_schedule": getattr(model_to_save, "_attnres_gamma_schedule", False),
                "attnres_gamma_start": getattr(model_to_save, "_attnres_gamma_start", None),
                "attnres_gamma_end": getattr(model_to_save, "_attnres_gamma_end", None),
                "attnres_gamma_ramp_frac": getattr(model_to_save, "_attnres_gamma_ramp_frac", None),
                "attnres_gamma_ramp_steps": getattr(model_to_save, "_attnres_gamma_ramp_steps", None),
                "distributed_backend": "fsdp" if is_fsdp_model(model) else "ddp" if dist.is_initialized() else "single",
                "optimizer_state_saved": optimizer_state_saved,
            }
            torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

            if export_attnres_state(model, tmp_ckpt_dir / "attnres_state.pt"):
                logging.info("Exported current AttnRes state to %s", tmp_ckpt_dir / "attnres_state.pt")

            # save norm stats
            norm_stats = data_config.norm_stats
            if norm_stats is not None and data_config.asset_id is not None:
                _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

            # Atomically move temp directory to final location
            if final_ckpt_dir.exists():
                shutil.rmtree(final_ckpt_dir)
            tmp_ckpt_dir.rename(final_ckpt_dir)

            logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

            # Log checkpoint to wandb
            if config.wandb_enabled:
                wandb.log({"checkpoint_step": global_step}, step=global_step)

        if dist.is_initialized():
            dist.barrier()


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def wrap_distributed_model(model, config: _config.TrainConfig, use_ddp: bool, world_size: int, device: torch.device):
    if not use_ddp:
        if config.fsdp_devices > 1:
            raise ValueError("FSDP requires launching with torchrun and WORLD_SIZE > 1")
        return model, "single"

    if config.fsdp_devices > 1:
        if config.fsdp_devices != world_size:
            logging.warning(
                "PyTorch AttnRes FSDP currently shards across the whole torchrun world. "
                "Got fsdp_devices=%d and world_size=%d; using world_size=%d.",
                config.fsdp_devices,
                world_size,
                world_size,
            )
        if config.pytorch_training_precision == "bfloat16":
            logging.info("Casting model parameters to uniform bfloat16 before root FSDP flattening")
            model.to(dtype=torch.bfloat16)
            mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
                cast_forward_inputs=False,
                cast_root_forward_inputs=False,
            )
        else:
            mixed_precision = None
        logging.info("Wrapping PI0Pytorch with FSDP FULL_SHARD across %d ranks", world_size)
        return (
            FSDP(
                model,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                device_id=device,
                use_orig_params=True,
            ),
            "fsdp",
        )

    logging.info("Wrapping PI0Pytorch with DDP across %d ranks", world_size)
    return (
        torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # AttnRes patches only PaliGemma's forward path.
            gradient_as_bucket_view=True,
            static_graph=world_size >= 8,
        ),
        "ddp",
    )


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and is_main and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    if dist.is_initialized():
        dist.barrier()

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        if is_main:
            exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
        if dist.is_initialized():
            dist.barrier()
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)

    # Log sample images to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # Build model
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    if isinstance(model_cfg, openpi.models.pi05_attnres_config.Pi05AttnResConfig):
        model = attnres_pytorch.PI05AttnResPytorch(
            model_cfg,
            num_blocks=model_cfg.attnres_num_blocks,
            adapter_rank=model_cfg.attnres_adapter_rank,
            no_adapter=model_cfg.attnres_no_adapter,
        ).to(device)
        logging.info("Created standalone PI05AttnResPytorch model")
    else:
        model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    # Load weights from weight_loader if specified (for fine-tuning)
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")

        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        if isinstance(model, attnres_pytorch.PI05AttnResPytorch):
            model.load_base_weights(model_path, device=device)
        else:
            safetensors.torch.load_model(model, model_path)
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    install_attnres_retrofit(model, device, model_cfg)

    model, distributed_backend = wrap_distributed_model(model, config, use_ddp, world_size, device)

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Create optimizer with config parameters
    trainable_params = get_trainable_parameters(model)
    if is_main:
        attnres_params = get_attnres_parameters(model, trainable_only=True)
        logging.info(
            "Trainable params: total_tensors=%d, attnres_tensors=%d, attnres_params=%.2fM",
            len(trainable_params),
            len(attnres_params),
            sum(param.numel() for param in attnres_params) / 1e6,
        )
    optim = torch.optim.AdamW(
        trainable_params,
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(f"Distributed backend: {distributed_backend}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    disable_tqdm = (not is_main) or (not sys.stderr.isatty())
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=disable_tqdm)
        if is_main
        else None
    )

    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            gamma_value = apply_attnres_gamma_schedule(model, global_step, config.num_train_steps)

            # The unified data loader returns (observation, actions) tuple
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901

            # Update LR
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # Forward pass
            losses = model(observation, actions)
            # Ensure losses is a tensor and handle different return types
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()

            # Backward pass
            loss.backward()
            sync_attnres_gradients(model)

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                get_trainable_parameters(model),
                max_norm=config.optimizer.clip_gradient_norm,
            )

            # Optimizer step
            optim.step()
            optim.zero_grad(set_to_none=True)

            # Clear gradients more aggressively
            extra_attnres_params = [] if attnres_is_registered(model) else get_attnres_parameters(model)
            for param in list(get_model_parameters(model)) + extra_attnres_params:
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            # Collect stats
            if is_main:
                infos.append(
                    {
                        "loss": loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "attnres_gamma": gamma_value,
                    }
                )

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                avg_gamma = None
                gamma_vals = [info["attnres_gamma"] for info in infos if info.get("attnres_gamma") is not None]
                if gamma_vals:
                    avg_gamma = sum(gamma_vals) / len(gamma_vals)
                gamma_suffix = f" gamma={avg_gamma:.4f}" if avg_gamma is not None else ""
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2e}{gamma_suffix} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e}{gamma_suffix} time={elapsed:.1f}s"
                )

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    if avg_gamma is not None:
                        log_payload["attnres_gamma"] = avg_gamma
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(model, optim, global_step, config, is_main, data_config)

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
