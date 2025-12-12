import os

import time
import argparse
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state, checkpoints
from flax.serialization import to_bytes

import wandb

from tom_nn import (
    AuxiliaryPredictorRNN, 
    passive_update, 
    build_passive_batch_from_sequences,
    PassiveTargets
)

from utils import (
    DIR_TO_IDX,
    get_direction_one_hot,
    NpzEpisodeDataset,
    pad_collate,
)


# ==========================================
# 2. TRAINING SETUP (JAX/FLAX)
# ==========================================

class TrainState(train_state.TrainState):
    pass

def create_train_state(rng, config):
    model = AuxiliaryPredictorRNN(
        num_actions=config['num_actions'],
        view_size=config['fov_size'],
        predict_frame=True,
        predict_action=False,
        obs_emb_dim=config['obs_emb_dim'],
        rnn_hidden_dim=config['rnn_hidden_dim']
    )

    dummy_obs = jnp.zeros((1, 1, config['fov_size'], config['fov_size'], 2), dtype=jnp.int32) 
    dummy_dir = jnp.zeros((1, 1, 4))
    dummy_act = jnp.zeros((1, 1), dtype=jnp.int32)
    dummy_rew = jnp.zeros((1, 1))
    
    dummy_inputs = {
        "obs_img": dummy_obs,
        "obs_dir": dummy_dir,
        "prev_action": dummy_act,
        "prev_reward": dummy_rew
    }
    
    h0 = model.initialize_carry(batch_size=1)
    variables = model.init(rng, dummy_inputs, h0)
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=config['lr'])
    )
    
    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    )

# ==========================================
# 3. MAIN LOOP
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,  default="../data/trajs/MiniGrid-ToM-TwoRoomsNoSwap-13x13")
    parser.add_argument("--work_dir", type=str, default="./checkpoints/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--fov_size", type=int, default=7)
    parser.add_argument("--num_actions", type=int, default=7) 
    parser.add_argument("--obs_emb_dim", type=int, default=16)
    parser.add_argument("--rnn_hidden_dim", type=int, default=1024)
    # WandB specific args
    parser.add_argument("--track", type=bool, default=True) # use wandb or not
    parser.add_argument("--wandb_project", type=str, default="aux-predictor-rnn")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Your wandb username/org")
    # save
    parser.add_argument("--save_every", type=int, default=10)
    
    args = parser.parse_args()
    if args.track:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"rnn_pred_{int(time.time())}"
        )

    # 1. Setup Data
    dataset = NpzEpisodeDataset(args.data_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=pad_collate, 
        num_workers=0, 
        drop_last=True
    )

    # 2. Setup JAX
    rng = jax.random.key(42)
    rng, init_rng = jax.random.split(rng)
    
    config = vars(args)
    state = create_train_state(init_rng, config)
    
    ckpt_dir = os.path.abspath(os.path.join(args.work_dir, "tomnets"))
    os.makedirs(ckpt_dir, exist_ok=True)
    
    @jax.jit
    def train_step(state, batch_inputs, batch_targets, init_hstate):
        return passive_update(
            state,
            init_hstate,
            inputs=batch_inputs,
            targets=batch_targets,
            view_size=args.fov_size,
            predict_frame=True,
            predict_action=False
        )

    # 3. Training Loop
    global_step = 0
    for epoch in range(args.epochs):
        epoch_losses = []
        start_time = time.time()
        
        for i, batch in enumerate(dataloader):
            obs_raw = jnp.array(batch['obs'])
            
            # Combine pad mask with done seq
            # mask_pad: 1=valid, 0=padded (Wait, logic in collate was: mask_pad=1 for valid)
            # done_seq: 1=episode ended.
            # build_passive_batch expects 'done_seq' where 1 means INVALID/TERMINAL.
            # So we treat padding (where mask_pad==0) as done.
            is_padded = 1.0 - jnp.array(batch['mask_pad'])
            eff_done = jnp.maximum(jnp.array(batch['done']), is_padded)

            inputs_jax, targets_jax = build_passive_batch_from_sequences(
                obs_seq=obs_raw,
                dir_seq=jnp.array(batch['dir']),
                prev_action_seq=jnp.array(batch['act']),
                prev_reward_seq=jnp.array(batch['rew']),
                next_frame_seq=obs_raw[..., 0], # Channel 0 for tile IDs
                next_other_action_seq=jnp.array(batch['next_act']),
                done_seq=eff_done
            )
            
            init_h = jnp.zeros((args.batch_size, 1, args.rnn_hidden_dim))
            
            # Update
            state, logs = train_step(state, inputs_jax, targets_jax, init_h)
            
            # Logging
            loss_val = logs['total_loss'].item()
            epoch_losses.append(loss_val)
            global_step += 1
            
            # --- [ADDED] WandB Logging ---
            if i % 10 == 0 and args.track:
                # Convert JAX scalars to Python floats for WandB
                wandb_log_dict = {
                    "train/total_loss": loss_val,
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                }
                # Add specific losses if they exist (frame_loss, action_loss)
                if 'frame_loss' in logs:
                    wandb_log_dict["train/frame_loss"] = logs['frame_loss'].item()
                if 'action_loss' in logs:
                    wandb_log_dict["train/action_loss"] = logs['action_loss'].item()
                
                wandb.log(wandb_log_dict)
                
                print(f"Ep {epoch} | It {i} | Loss: {loss_val:.4f}", end="\r")
        
        avg_loss = np.mean(epoch_losses)
        print(f"\nEpoch {epoch} Finished | Avg Loss: {avg_loss:.4f} | Time: {time.time()-start_time:.2f}s")

        # Save Checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_name = f"checkpoint_{epoch}.msgpack"
            save_path = os.path.join(ckpt_dir, ckpt_name)
            with open(save_path, "wb") as f:
                f.write(to_bytes(state))
            print(f"Saved checkpoint to: {save_path}")

    if args.track:
        wandb.finish()

if __name__ == "__main__":
    main()