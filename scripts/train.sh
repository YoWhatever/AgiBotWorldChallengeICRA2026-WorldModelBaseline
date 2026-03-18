#!/usr/bin/env bash

name="agibotworld"
tag="_expv0"
config_file=${1:-configs/${name}/train_config.yaml}

save_root="./log"

seed=123

if [ -z "$NGPU" ]; then
  NGPU=$(python3 - <<'PY'
import os
import torch

def _mod_available(name: str) -> bool:
    mod = getattr(torch, name, None)
    if mod is None:
        return False
    fn = getattr(mod, "is_available", None)
    return bool(fn()) if callable(fn) else False

backend = (os.getenv("ABW_BACKEND") or os.getenv("TORCH_DEVICE") or "").strip().lower()
if backend in ("gpu", "cuda"):
    backend = "cuda"

count = 0
if backend:
    mod = getattr(torch, backend, None)
    if mod is not None and hasattr(mod, "device_count"):
        try:
            count = int(mod.device_count())
        except Exception:
            count = 0
elif _mod_available("cuda"):
    count = torch.cuda.device_count()
elif _mod_available("xpu"):
    count = torch.xpu.device_count()
elif _mod_available("npu"):
    count = torch.npu.device_count()
elif _mod_available("mlu"):
    count = torch.mlu.device_count()
elif _mod_available("musa"):
    count = torch.musa.device_count()

print(count or 1)
PY
  )
fi
export OMP_NUM_THREADS=4

echo "Training on 1 Node, $NGPU GPUs"
echo $config_file


torchrun --nnodes=1 \
    --nproc_per_node=$NGPU \
    --node_rank=0 \
    trainer/trainer.py \
    --base $config_file \
    --train \
    --seed $seed \
    --name ${name}${tag} \
    --logdir $save_root \
    --devices $NGPU \
    lightning.trainer.num_nodes=1

# echo $?
