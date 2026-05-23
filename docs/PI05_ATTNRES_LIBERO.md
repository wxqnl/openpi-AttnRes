# PI0.5 AttnRes LIBERO 使用说明

本文档说明 openpi 内部维护的独立模型 `pi05_attnres_libero` 的用途、训练、checkpoint 格式和 LIBERO 评测方式。

## 1. 模型定位

`pi05_attnres_libero` 是一个独立的 PyTorch VLA 模型，不再依赖“先加载 pi05，再额外挂 AttnRes sidecar”的临时流程。

核心代码：

- `src/openpi/models_pytorch/attnres.py`
  - `PI05AttnResPytorch`
  - `PI05PaliGemmaAttnResRetrofit`
  - AttnRes router / adapter / gamma
- `src/openpi/models/pi05_attnres_config.py`
  - `Pi05AttnResConfig`
- `src/openpi/training/config.py`
  - 注册名：`pi05_attnres_libero`

新模型的 `state_dict` 结构是：

```text
pi0.*       # 原 pi0.5 VLA 主体，包括 PaliGemma、action expert、action head
attnres.*   # AttnRes router / adapters / gamma
```

因此新 checkpoint 的 `model.safetensors` 本身就是完整模型，不需要额外传 `attnres_state.pt` 才能推理。

## 2. 环境和数据

推荐在 openpi 根目录执行：

```bash
cd /home/user01/Minko/openpi
source .venv/bin/activate
export PYTHONPATH=$PWD/src
```

LIBERO 数据和 Hugging Face cache 建议放在：

```bash
export HF_HOME=/home/user01/Minko/datasets/huggingface
export HF_HUB_CACHE=/home/user01/Minko/datasets/huggingface/hub
export HF_DATASETS_CACHE=/home/user01/Minko/datasets/huggingface/datasets
export HF_LEROBOT_HOME=/home/user01/Minko/datasets/huggingface/lerobot
unset LEROBOT_HOME
```

base PyTorch 权重默认路径：

```text
./models/pi05_libero_pytorch
```

如果新机器没有这个目录，先用 `scripts/prepare_pi05_base_pytorch.py` 准备 pi0.5 base PyTorch ckpt。

## 3. 训练新模型

推荐入口：

```bash
CUDA_VISIBLE_DEVICES=2,3 \
EXP_NAME=pi05_attnres_libero_ramp30k_b32 \
PYTORCH_WEIGHT_PATH=./models/pi05_libero_pytorch \
scripts/run_pi05_attnres_libero_train.sh \
  --no-wandb-enabled \
  --overwrite
```

默认配置：

- config：`pi05_attnres_libero`
- 全局 batch size：`32`
- 两卡 FSDP：`--fsdp-devices 2`
- 每卡 batch：`16`
- 总步数：`30000`
- 保存间隔：`1000`
- AttnRes blocks：`9`
- adapter rank：`256`
- AttnRes 初始化：`random`
- gamma schedule：从 `0.0` 线性升到 `1.0`
- gamma ramp steps：`30000`

### 推荐 recipe：gamma_ramp10k

LIBERO 4-suite 上经验最佳的 recipe 是 gamma 在前 **10000 步**就升到 1.0（而不是 dataclass 默认的 30000）。
ramp10k 在 LIBERO 上 30k 训练步后 4-suite 平均 96.85%，30k ramp 默认配置略低。
直接用现成入口：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
EXP_NAME=pi05_attnres_gamma_ramp10k_b32_fsdp_2gpu \
PYTORCH_WEIGHT_PATH=./models/pi05_libero_pytorch \
scripts/run_pi05_attnres_libero_train_gamma_ramp10k.sh \
  --no-wandb-enabled \
  --overwrite
```

这个脚本会显式设置 `OPENPI_ATTNRES_GAMMA_RAMP_STEPS=10000`（可以通过 `GAMMA_RAMP_STEPS=20000 ...` 覆盖）。其它推荐写法看脚本顶部注释。

等价展开命令：

```bash
CUDA_VISIBLE_DEVICES=2,3 ./.venv/bin/python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  scripts/train_pytorch_pi05_attnres.py pi05_attnres_libero \
  --exp-name pi05_attnres_libero_ramp30k_b32 \
  --pytorch-weight-path ./models/pi05_libero_pytorch \
  --batch-size 32 \
  --num-train-steps 30000 \
  --save-interval 1000 \
  --log-interval 100 \
  --fsdp-devices 2 \
  --no-wandb-enabled \
  --overwrite
```

常用改法：

```bash
# 只跑 1000 steps smoke test
scripts/run_pi05_attnres_libero_train.sh \
  --exp-name pi05_attnres_smoke_1000 \
  --num-train-steps 1000 \
  --save-interval 1000 \
  --no-wandb-enabled \
  --overwrite

# 改 batch size
scripts/run_pi05_attnres_libero_train.sh \
  --batch-size 64 \
  --no-wandb-enabled \
  --overwrite

# 改 gamma ramp
scripts/run_pi05_attnres_libero_train.sh \
  --model.attnres-gamma-ramp-steps 60000 \
  --no-wandb-enabled \
  --overwrite
```

## 4. Checkpoint 格式

新模型 checkpoint 路径：

```text
checkpoints/pi05_attnres_libero/<exp_name>/<step>/
```

目录内容：

```text
model.safetensors   # 完整 standalone 模型：pi0.* + attnres.*
metadata.pt
assets/
```

`attnres_state.pt` 对新模型不是必需文件。训练脚本可能仍会额外导出它用于兼容旧流程，但标准加载和评测只依赖 `model.safetensors`。

## 5. 启动评测 server

推荐入口：

```bash
CUDA_VISIBLE_DEVICES=2 \
scripts/run_pi05_attnres_libero_eval_server.sh \
  checkpoints/pi05_attnres_libero/pi05_attnres_libero_ramp30k_b32/30000 \
  18090
```

等价展开命令：

```bash
CUDA_VISIBLE_DEVICES=2 ./.venv/bin/python scripts/serve_policy.py \
  --port 18090 \
  policy:checkpoint \
  --policy.config pi05_attnres_libero \
  --policy.dir checkpoints/pi05_attnres_libero/pi05_attnres_libero_ramp30k_b32/30000
```

注意：这里使用的是标准 `serve_policy.py`，不是旧的 `serve_pi05_attnres_libero_policy.py`。因为 `pi05_attnres_libero` 已经是独立模型，加载 checkpoint 时会自动恢复 AttnRes。

## 6. 跑 LIBERO eval

server 启动后，在另一个终端跑 LIBERO client。**推荐用 GPU 加速渲染（MUJOCO_GL=egl）**，比 osmesa 快 5-10×：

```bash
export PYTHONPATH=$PWD/third_party/libero:$PWD/packages/openpi-client/src
export LIBERO_CONFIG_PATH=/tmp/libero
export MUJOCO_GL=egl
export EGL_DEVICE_ID=0
export CUDA_VISIBLE_DEVICES=1   # 用空闲卡做 EGL 渲染；跟 server 卡分开

./examples/libero/.venv/bin/python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 18090 \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 1 \
  --args.video-out-path outputs/libero_rollout/pi05_attnres_spatial/videos
```

如果机器没有 EGL，可以退回 CPU 渲染（慢，仅用作 fallback）：

```bash
export MUJOCO_GL=osmesa
export CUDA_VISIBLE_DEVICES=
```

只跑某一个 task：

```bash
/home/user01/Minko/openpi/examples/libero/.venv/bin/python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 18090 \
  --args.task-suite-name libero_spatial \
  --args.task-start 0 \
  --args.task-end 1 \
  --args.num-trials-per-task 1 \
  --args.video-out-path outputs/libero_rollout/pi05_attnres_spatial_task0/videos
```

常用 suite：

```text
libero_spatial
libero_object
libero_goal
libero_10
libero_90
```

## 6.1 LIBERO-Pro 评测结果（libero_10 row, swap / task / lan）

[LIBERO-Pro](https://github.com/Zxy-MLlab/LIBERO-PRO)（arXiv [2510.03827](https://arxiv.org/abs/2510.03827)）
是 LIBERO 的扰动评测扩展，在 4 个 base suite × 5 个扰动维度（Object / Position(swap) / Semantic(lan) / Task / Environment）
共 20 个 cell 上测 VLA 的真实泛化能力。HF 数据集 `zhouxueyang/LIBERO-Pro` 当前只
ship 了 4 个维度的扰动 BDDL/init 文件（缺 env）。

我们在 `libero_10` row 上跑了 4 个 ckpt 的对照。**Object 维度的结果暂未列入**（涉及
跨品类替换，正在分析中）；下表是 swap / task / lan 三维：

| ckpt                          | swap (Pos) | task   | lan (Sem) | 3-dim avg |
|-------------------------------|-----------:|-------:|----------:|----------:|
| paper pi0.5 (leaderboard)     |      0.08  |  0.01  |    0.93   |   0.340   |
| our base_20k                  |      0.106 |  0.062 |    0.918  |   0.362   |
| our base_30k                  |      0.094 |  0.066 |    0.916  |   0.359   |
| **our gamma_ramp10k_20k**     |    **0.134**| 0.096 |  **0.952**|  **0.394**|
| our gamma_ramp10k_30k         |      0.098 |**0.100**|   0.932  |   0.377   |

每个 cell 都是 500 episodes（10 task × 50 trial）的成功率。要点：

- **同步数下 AttnRes vs base：** ramp10k_20k vs base_20k 在 swap/task/lan 上分别 +2.8 / +3.4 / +3.4 pp（3-dim avg +3.2 pp）；ramp10k_30k vs base_30k 在 task 上 +3.4 pp，其它两维近持平。
- **AttnRes 在 task perturbation 上稳定 +3.4 pp**——LIBERO-Pro 主要 stress-test 的就是 task / position 这两维（paper 表中 pi0.5 task=0.01、swap=0.08，崩得最严重），AttnRes 在 task 上把 0.01 拉到 0.10，是数量级提升。
- **20k 比 30k 更鲁棒**（ramp10k_20k 3-dim avg 0.394 > ramp10k_30k 的 0.377），表明继续训到 30k 在扰动维度上略 overfit。
- **复现 paper：** 我们 base 在 task 上比 paper pi0.5 高（0.066 vs 0.01），可能源自 HF 数据集 2025-11-05 重新冻结后扰动种子的差异，paper leaderboard 的版本与当前 HF 文件未必完全一致。

LIBERO-Pro 评测的搭建：克隆 [LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO) repo，
下载 HF dataset 把 bddl_files / init_files merge 进 `libero/libero/{bddl_files,init_files}/`，
然后用 `examples/libero/main.py --args.task-suite-name libero_10_swap`（等）启动。
完整 setup 不在 openpi 本仓库内维护，看 LIBERO-PRO 的 README。

## 7. 旧 checkpoint 迁移

早期实验 checkpoint 是旧格式：

```text
model.safetensors   # 只有 pi0.5 VLA 主体
attnres_state.pt    # AttnRes sidecar
metadata.pt
assets/
```

可以转换成新 standalone 格式：

```bash
PYTHONPATH=src ./.venv/bin/python scripts/convert_pi05_attnres_checkpoint.py \
  --source-ckpt checkpoints/pi05_libero/<old_exp>/<step> \
  --output-ckpt checkpoints/pi05_attnres_libero/<new_exp>/<step> \
  --config pi05_attnres_libero \
  --device cpu \
  --overwrite
```

转换后用新方式评测：

```bash
CUDA_VISIBLE_DEVICES=2 \
scripts/run_pi05_attnres_libero_eval_server.sh \
  checkpoints/pi05_attnres_libero/<new_exp>/<step> \
  18090
```

## 8. 设计细节

AttnRes 只作用在 pi0.5 的 PaliGemma VLM language model 分支上。训练 LIBERO 时，action head 和 VLA 主体一起在 `PI05AttnResPytorch` 里维护，AttnRes 作为注册子模块保存。

gamma curriculum：

```text
step 0      gamma = 0
step 30000  gamma = 1
```

gamma 为 0 时，AttnRes wrapper 应当等价于 base PaliGemma forward。当前实现已修复 prefix KV cache 路径，避免 gamma=0 时破坏 VLA denoise 推理。

## 9. 文件索引

训练：

```text
scripts/run_pi05_attnres_libero_train.sh
scripts/train_pytorch_pi05_attnres.py
```

模型：

```text
src/openpi/models/pi05_attnres_config.py
src/openpi/models_pytorch/attnres.py
```

评测：

```text
scripts/run_pi05_attnres_libero_eval_server.sh
scripts/serve_policy.py
examples/libero/main.py
```

迁移：

```text
scripts/convert_pi05_attnres_checkpoint.py
```

旧兼容入口：

```text
scripts/serve_pi05_attnres_libero_policy.py
```

旧兼容入口只用于加载历史 sidecar checkpoint。新实验应使用 `pi05_attnres_libero` 和标准 `serve_policy.py`。

## 10. 常见问题

### 训练只允许用指定 GPU

显式设置：

```bash
CUDA_VISIBLE_DEVICES=2,3 scripts/run_pi05_attnres_libero_train.sh ...
```

### eval server 只用一张卡

```bash
CUDA_VISIBLE_DEVICES=2 scripts/run_pi05_attnres_libero_eval_server.sh <ckpt> 18090
```

### LIBERO client 不要占 GPU

```bash
CUDA_VISIBLE_DEVICES= /home/user01/Minko/openpi/examples/libero/.venv/bin/python examples/libero/main.py ...
```

### 如何确认 checkpoint 是新格式

```bash
PYTHONPATH=src ./.venv/bin/python - <<'PY'
from safetensors import safe_open
p = "checkpoints/pi05_attnres_libero/<exp>/<step>/model.safetensors"
with safe_open(p, framework="pt", device="cpu") as f:
    keys = list(f.keys())
print(any(k.startswith("pi0.") for k in keys))
print(any(k.startswith("attnres.") for k in keys))
PY
```

两个输出都应为 `True`。
