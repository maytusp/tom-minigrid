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
from flax.training import train_state
from flax.serialization import to_bytes

import wandb

from .tom_nn import (
    AuxiliaryPredictorRNN, 
    passive_update, 
    build_passive_batch_from_sequences,
    PassiveTargets
)

from .utils import (
    NpzEpisodeDataset,
    pad_collate,
    crop_fov_from_allocentric_rgb,
    crop_fov_symbolic_allocentric,
)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,  default="./home/maytus/tom/tom-minigrid/logs/train_trajs/MiniGrid-Protagonist-TwoRoomsNoSwap-9x9vs9-swap")
    parser.add_argument("--work_dir", type=str, default="./checkpoints/observers/train-env-MiniGrid-Protagonist-ProcGen-9x9vs9/staticWeight_mask")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--fov_size", type=int, default=9)
    parser.add_argument("--observer_r", type=int, default=9) # observer row position
    parser.add_argument("--observer_c", type=int, default=5) # observer column position
    parser.add_argument("--observer_d", type=int, default=0) # observer direction

    parser.add_argument("--num_actions", type=int, default=6) 
    parser.add_argument("--obs_emb_dim", type=int, default=16)
    parser.add_argument("--rnn_hidden_dim", type=int, default=256)
    
    # Loss weighting args
    parser.add_argument("--static_weight", type=float, default=0.01, 
                        help="Weight for static pixels. 1.0 = standard, 0.0 = strict mask, 0.1 = balanced.")

    # WandB specific args
    parser.add_argument("--track", type=bool, default=True) 
    parser.add_argument("--wandb_project", type=str, default="tom_observer_training")
    parser.add_argument("--wandb_entity", type=str, default=None)
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
    
    ckpt_dir = args.work_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # JIT the update step, baking in the static_weight
    @jax.jit
    def train_step(state, batch_inputs, batch_targets, init_hstate):
        return passive_update(
            state,
            init_hstate,
            inputs=batch_inputs,
            targets=batch_targets,
            view_size=args.fov_size,
            predict_frame=True,
            predict_action=False,
            static_pixel_weight=args.static_weight  # <--- Passed here
        )

    @jax.jit
    def batch_crop_fov(allo_obs):
        return jax.vmap(jax.vmap(
            lambda o: crop_fov_symbolic_allocentric(
                grid_sym=o, 
                r=args.observer_r,
                c=args.observer_c, 
                view_size=args.fov_size, 
                dir_id=args.observer_d # up
            )
        ))(allo_obs)
        
    # Training Loop
    global_step = 0
    for epoch in range(args.epochs):
        epoch_losses = []
        start_time = time.time()
        
        for i, batch in enumerate(dataloader):
            # Load raw allocentric data [B, T, 11, 11, 2]
            obs_raw = jnp.array(batch['obs']) 

            # --- 1. CROP FRAMES ---
            # Convert 11x11 Allocentric -> 9x9 Egocentric
            obs_cropped = batch_crop_fov(obs_raw)
            
            # --- 2. PREPARE INPUTS (Use Cropped Data) ---
            # Input: Time t
            obs_inputs = obs_cropped[:, :-1]
            

            
            act_inputs = jnp.array(batch['act'])[:, :-1]
            rew_inputs = jnp.array(batch['rew'])[:, :-1]
            
            # --- 3. PREPARE TARGETS (Use Cropped Data) ---
            # Target: Time t+1
            # We only need the Tile ID channel (0) for labels
            obs_targets_labels = obs_cropped[:, 1:, ..., 0] 
            next_act_targets = jnp.array(batch['next_act'])[:, :-1]

            # --- 4. CALCULATE CHANGE MASK ---
            # Detect changes in the *Cropped* view
            # If a pixel changed in the world but is outside the 9x9 view, we don't care.
            is_diff = (obs_cropped[:, 1:] != obs_cropped[:, :-1])
            change_mask = jnp.any(is_diff, axis=-1).astype(jnp.float32)

            # --- 5. HANDLE DONE / PADDING ---
            mask_pad = jnp.array(batch['mask_pad'])[:, :-1]
            is_padded = 1.0 - mask_pad
            done_seq = jnp.array(batch['done'])[:, :-1]
            eff_done = jnp.maximum(done_seq, is_padded)

            inputs_jax, targets_jax = build_passive_batch_from_sequences(
                obs_seq=obs_inputs,
                prev_action_seq=act_inputs,
                prev_reward_seq=rew_inputs,
                next_frame_seq=obs_targets_labels, 
                next_other_action_seq=next_act_targets,
                done_seq=eff_done,
                spatial_mask_seq=change_mask 
            )
            
            init_h = jnp.zeros((args.batch_size, 1, args.rnn_hidden_dim))
            
            state, logs = train_step(state, inputs_jax, targets_jax, init_h)
            
            # ... (Logging logic) ...
            loss_val = logs['total_loss'].item()
            epoch_losses.append(loss_val)
            global_step += 1
            
            if i % 10 == 0 and args.track:
                wandb_log_dict = {
                    "train/total_loss": loss_val,
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                }
                if 'frame_loss' in logs:
                    wandb_log_dict["train/frame_loss"] = logs['frame_loss'].item()
                    wandb_log_dict["train/frame_loss_dynamic"] = logs['frame_loss_dynamic'].item()
                    wandb_log_dict["train/frame_loss_static"] = logs['frame_loss_static'].item()
                if 'action_loss' in logs:
                    wandb_log_dict["train/action_loss"] = logs['action_loss'].item()
                
                wandb.log(wandb_log_dict)
                print(f"Ep {epoch} | It {i} | Loss: {loss_val:.4f}", end="\r")
        
        avg_loss = np.mean(epoch_losses)
        print(f"\nEpoch {epoch} Finished | Avg Loss: {avg_loss:.4f} | Time: {time.time()-start_time:.2f}s")

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