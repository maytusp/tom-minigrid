#!/bin/bash --login
#SBATCH -p gpuL               # A100 GPUs
#SBATCH -G 1                  # 1 GPU
#SBATCH -t 1-0                # Wallclock limit (1-0 is 1 day, 4-0 is the max permitted)
#SBATCH -n 1                  # One Slurm task
#SBATCH -c 12                  # 8 CPU cores available to the host code.
                              # Can use up to 12 CPUs with an A100 GPU.
                              # Can use up to 12 CPUs with an L40s GPU.

# Latest version of CUDA

source activate jax

# FB
python -m training.eval_pred_act --model_type dual_perspective --env_id MiniGrid-ToM-TwoRoomsSwap-9x9vs9 --checkpoint ./checkpoints/observers/tworoom-noswap/DualNet-FreezeFPnet_lr0.001_bs128_seed1/checkpoint_49.msgpack --vid_out_dir logs/eval_pred_action/DualNet-FromScratch/swap_delay1


# python -m training.eval_pred_act --model_type dual_perspective --env_id MiniGrid-ToM-TwoRoomsSwap-9x9vs9 --checkpoint ./checkpoints/observers/tworoom-noswap/DualNet-FinetuneFPnet_lr0.001_bs128_seed1/checkpoint_49.msgpack --vid_out_dir logs/eval_pred_action/DualNet-FromScratch/swap_delay1

# python -m training.eval_pred_act --model_type dual_perspective --env_id MiniGrid-ToM-TwoRoomsSwap-9x9vs9 --checkpoint ./checkpoints/observers/tworoom-noswap/DualNet-FromScratch_lr0.001_bs128_seed1/checkpoint_49.msgpack --vid_out_dir logs/eval_pred_action/DualNet-FromScratch/swap_delay1


# TB Delay 4
python -m training.eval_pred_act --model_type dual_perspective --env_id MiniGrid-ToM-TwoRoomsSwap-9x9vs9-d4 --checkpoint ./checkpoints/observers/tworoom-noswap/DualNet-FreezeFPnet_lr0.001_bs128_seed1/checkpoint_49.msgpack --vid_out_dir logs/eval_pred_action/DualNet-FromScratch/swap_delay4 

# python -m training.eval_pred_act --model_type dual_perspective --env_id MiniGrid-ToM-TwoRoomsSwap-9x9vs9-d4 --checkpoint ./checkpoints/observers/tworoom-noswap/DualNet-FinetuneFPnet_lr0.001_bs128_seed1/checkpoint_49.msgpack --vid_out_dir logs/eval_pred_action/DualNet-FromScratch/swap_delay4 

# python -m training.eval_pred_act --model_type dual_perspective --env_id MiniGrid-ToM-TwoRoomsSwap-9x9vs9-d4 --checkpoint ./checkpoints/observers/tworoom-noswap/DualNet-FromScratch_lr0.001_bs128_seed1/checkpoint_49.msgpack --vid_out_dir logs/eval_pred_action/DualNet-FromScratch/swap_delay4 


# TB Delay 8

python -m training.eval_pred_act --model_type dual_perspective --env_id MiniGrid-ToM-TwoRoomsSwap-9x9vs9-d8 --checkpoint ./checkpoints/observers/tworoom-noswap/DualNet-FreezeFPnet_lr0.001_bs128_seed1/checkpoint_49.msgpack --vid_out_dir logs/eval_pred_action/DualNet-FromScratch/swap_delay8

# python -m training.eval_pred_act --model_type dual_perspective --env_id MiniGrid-ToM-TwoRoomsSwap-9x9vs9-d8 --checkpoint ./checkpoints/observers/tworoom-noswap/DualNet-FinetuneFPnet_lr0.001_bs128_seed1/checkpoint_49.msgpack --vid_out_dir logs/eval_pred_action/DualNet-FromScratch/swap_delay8

# python -m training.eval_pred_act --model_type dual_perspective --env_id MiniGrid-ToM-TwoRoomsSwap-9x9vs9-d8 --checkpoint ./checkpoints/observers/tworoom-noswap/DualNet-FromScratch_lr0.001_bs128_seed1/checkpoint_49.msgpack --vid_out_dir logs/eval_pred_action/DualNet-FromScratch/swap_delay8 


