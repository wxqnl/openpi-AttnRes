import dataclasses
import logging
import pathlib

from safetensors import safe_open
import safetensors.torch
from typing_extensions import override

from openpi.models import pi0_config

logger = logging.getLogger("openpi")


@dataclasses.dataclass(frozen=True)
class Pi05AttnResConfig(pi0_config.Pi0Config):
    """pi0.5 config whose PyTorch model owns PaliGemma AttnRes modules."""

    pi05: bool = True
    action_horizon: int = 10
    discrete_state_input: bool = False

    attnres_num_blocks: int = 9
    attnres_adapter_rank: int = 256
    attnres_no_adapter: bool = False
    attnres_init: str = "random"
    attnres_trainable: bool = True
    attnres_gamma_schedule: bool = True
    attnres_gamma_start: float = 0.0
    attnres_gamma_end: float = 1.0
    attnres_gamma_ramp_frac: float = 1.0
    attnres_gamma_ramp_steps: int | None = 30_000
    attnres_state_filename: str = "attnres_state.pt"

    @override
    def load_pytorch(self, train_config, weight_path: str):
        from openpi.models_pytorch.attnres import PI05AttnResPytorch

        logger.info("Loading standalone PI05AttnResPytorch from %s", weight_path)
        model = PI05AttnResPytorch(
            self,
            num_blocks=self.attnres_num_blocks,
            adapter_rank=self.attnres_adapter_rank,
            no_adapter=self.attnres_no_adapter,
        )
        with safe_open(weight_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
        has_pi0_prefix = any(key.startswith("pi0.") for key in keys)
        has_attnres_tensors = any(key.startswith("attnres.") for key in keys)
        checkpoint_dir = pathlib.Path(weight_path).parent
        state_path = checkpoint_dir / self.attnres_state_filename

        if has_pi0_prefix:
            missing, unexpected = safetensors.torch.load_model(model, weight_path, strict=False)
            allowed_tied_keys = {"pi0.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"}
            allowed_missing = set(allowed_tied_keys)
            if not has_attnres_tensors:
                allowed_missing.update(key for key in missing if key.startswith("attnres."))
            disallowed_missing = sorted(set(missing) - allowed_missing)
            disallowed_unexpected = sorted(set(unexpected) - allowed_tied_keys)
            if disallowed_missing or disallowed_unexpected:
                raise RuntimeError(
                    "Error(s) in loading state_dict for PI05AttnResPytorch: "
                    f"missing={disallowed_missing}, unexpected={disallowed_unexpected}"
                )
            if missing or unexpected:
                logger.info(
                    "Ignored tied embedding keys while loading PI05AttnRes checkpoint: %s",
                    sorted(allowed_tied_keys),
                )
        else:
            model.load_base_weights(weight_path)
            has_attnres_tensors = False

        # Backward-compatible migration path for older checkpoints that saved
        # AttnRes separately. New checkpoints should already include attnres.*
        # tensors in model.safetensors and will not need this file.
        if not has_attnres_tensors:
            if not state_path.exists():
                raise FileNotFoundError(
                    f"Standalone PI05AttnRes checkpoint is missing attnres.* tensors and no fallback "
                    f"{self.attnres_state_filename} exists in {checkpoint_dir}"
                )
            model.load_attnres_state(str(state_path))
            logger.info("Loaded legacy sidecar AttnRes state from %s", state_path)
        return model
