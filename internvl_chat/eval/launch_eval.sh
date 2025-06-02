set -x

MASTER_PORT=${MASTER_PORT:-63665}
PORT=${PORT:-63665}
GPUS=${GPUS:-8}
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

### Action Verification Task

# Set the model path in --checkpoint
# Set the jsonl data path in --data_path
# Set the GUI_Odyssey image folder in --image_root

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  -m eval_tasks.reflection_action_verification_infer \
  --checkpoint "craigwu/GUI_Reflection_8b_pretrain" \
  --data_path  "path_to_action_verification.jsonl" \
  --image_root "path_to_image_folder_of_GUI_Odyssey" \
  --max_num 6


### Action Reversal Task

# Set the model path in --checkpoint
# Set the jsonl data path in --data_path
# Set the GUI_Odyssey image folder in --image_root_odyssey
# Set the AndroidControl image folder in --image_root_ac

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  -m eval_tasks.reflection_action_reversal_infer \
  --checkpoint "craigwu/GUI_Reflection_8b_pretrain" \
  --data_path  "path_to_action_reversal.jsonl" \
  --image_root_odyssey "path_to_image_folder_of_GUI_Odyssey" \
  --image_root_ac "path_to_image_folder_of_Android_Control" \
  --max_num 6


### Mistake-informed Reattempt Task

# Set the model path in --checkpoint
# Set the jsonl data path in --data_path
# Set the screenspot/screenspot_v2 image folder in --image_root

# set --inference_round to 1 for the first attempt, for the subsequent attempts, change the --data_path to the result file path of last attempt and increment the value of --inference_round 

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  -m eval_tasks.reflection_reattempt_infer \
  --checkpoint "craigwu/GUI_Reflection_8b_pretrain" \
  --data_path  "path_to_reattempt_screenspot(_v2)" \
  --image_root "path_to_image_folder_of_screenspot(_v2)" \
  --inference_round 1 \
  --max_num 12