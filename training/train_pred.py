import os
import time
import argparse
from typing import Dict, Any

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax.serialization import to_bytes

import wandb
from torch.utils.data import DataLoader

# Import your modules
from .tom_nn import (
    create_model,
    build_passive_batch_from_sequences,
    PassiveTargets
)
from .utils import (
    NpzEpisodeDataset,
    pad_collate
)
from xminigrid.core.constants import AGENT_IDS

class TrainState(train_state.TrainState):
    pass
import jax.numpy as jnp

def extract_state_indices(obs_inputs, agent_ids):
    """
    Extracts the 1D state index of the agent from the observation tensor.
    
    Args:
        obs_inputs: Array of shape [B, T, H, W, C].
        agent_ids: List of integers representing the agent in channel 0.
                    
    Returns:
        state_indices: Integer array of shape [B, T] containing the flattened 1D index.
    """
    B, T, H, W, C = obs_inputs.shape
    id_channel = obs_inputs[..., 0]  # Shape: [B, T, H, W]
    
    # Convert the Python list to a JAX array
    agent_ids_array = jnp.array(agent_ids, dtype=id_channel.dtype)
    
    # Create a boolean mask: True wherever the id_channel matches ANY of the agent_ids
    is_agent = jnp.isin(id_channel, agent_ids_array)
    
    # Flatten the spatial dimensions -> Shape: [B, T, H * W]
    flat_is_agent = is_agent.reshape(B, T, -1)
    
    # Find the 1D index of the agent
    state_indices = jnp.argmax(flat_is_agent, axis=-1)  # Shape: [B, T]
    
    return state_indices

def compute_empirical_sr(state_indices, done_seq, num_states, gammas=[0.5, 0.9, 0.99]):
    """
    Computes the normalized empirical SR targets.
    state_indices: [B, T] integer array of agent positions (flattened 1D indices).
    done_seq: [B, T] float array indicating episode termination.
    """
    B, T = state_indices.shape
    # Indicator function I(s_{t+dt} = s)
    one_hot_states = jax.nn.one_hot(state_indices, num_states) # [B, T, Num_States]
    
    # We scan backwards through time to compute the discounted sum
    def scan_fn(carry, inputs):
        one_hot_s, done = inputs
        new_carry = {}
        sr_t = []
        for g in gammas:
            # If done, we don't carry over future occupancy
            val = one_hot_s + g * carry[g] * (1.0 - done[:, None])
            new_carry[g] = val
            sr_t.append(val)
        return new_carry, jnp.stack(sr_t, axis=-1)

    # Initialize carry dictionaries
    init_carry = {g: jnp.zeros((B, num_states)) for g in gammas}
    
    # Time-major inputs for lax.scan
    inputs = (jnp.swapaxes(one_hot_states, 0, 1), jnp.swapaxes(done_seq, 0, 1))
    
    _, sr_unnormalized = jax.lax.scan(scan_fn, init_carry, inputs, reverse=True)
    
    # Swap back to batch-major [B, T, Num_States, Num_Gammas]
    sr_unnormalized = jnp.swapaxes(sr_unnormalized, 0, 1)
    
    # Normalize by Z so sum over states is 1
    Z = jnp.sum(sr_unnormalized, axis=2, keepdims=True)
    sr_normalized = sr_unnormalized / jnp.maximum(Z, 1e-8)
    
    return sr_normalized

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./logs/train_trajs/tworoom_noswap")
    parser.add_argument("--work_dir", type=str, default="./checkpoints/observers/tworoom-noswap/tp-sr/")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    
    # Model Config
    parser.add_argument("--model_type", type=str, default="third_person", 
                        choices=["third_person", "dual_perspective"])
    parser.add_argument("--p_checkpoint", type=str, default="./checkpoints/MiniGrid-ToM-TwoRoomsNoSwap-9x9vs9/MiniGrid-ToM-TwoRoomsNoSwap-9x9vs9-ppo_final.msgpack",
                        help="Path to protagonist checkpoint")
    
    # Architecture Args (Must match P's training config)
    parser.add_argument("--num_actions", type=int, default=6) 
    parser.add_argument("--fp_emb", type=int, default=16)
    parser.add_argument("--fp_rnn", type=int, default=256)
    parser.add_argument("--tp_emb", type=int, default=16)
    parser.add_argument("--tp_rnn", type=int, default=256)

    # Logging
    parser.add_argument("--track", action="store_true", default=True) 
    parser.add_argument("--wandb_project", type=str, default="tom_observer_training")
    parser.add_argument("--save_every", type=int, default=10)

    parser.add_argument("--use_sr", action="store_true", default=True, 
                            help="Enable Successor Representation prediction and loss.")
    parser.add_argument("--sr_coef", type=float, default=1.0, 
                        help="Weight multiplier for the SR loss.")
    parser.add_argument("--num_states", type=int, default=81, 
                        help="Total number of discrete states for the SR (update based on your grid size).")
    args = parser.parse_args()

    if args.track:
        wandb.init(project=args.wandb_project, config=vars(args))

    # 1. Setup Data
    dataset = NpzEpisodeDataset(args.data_dir)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, 
        collate_fn=pad_collate, num_workers=0, drop_last=True
    )

    # 2. Setup Model
    rng = jax.random.key(args.seed)
    config = vars(args)
    
    # create_model handles the dual/third switch and P-loading
    model, params = create_model(args.model_type, config, rng)
    
    tx = optax.adam(args.lr)
    state = TrainState.create(apply_fn=model.apply, params=params['params'], tx=tx)

    @jax.jit
    def train_step(state, inputs_fp, inputs_tp, h_fp, h_tp, targets: PassiveTargets, target_sr):
        
        def loss_fn(params, *, use_focal_loss: bool = True, focal_gamma: float = 5.0, focal_eps: float = 1e-8):
            logits, pred_sr, _, _ = state.apply_fn(
                {'params': params},
                inputs_fp, h_fp, inputs_tp, h_tp
            )

            # --- Action Loss ---
            A = logits.shape[-1]
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            probs = jnp.exp(log_probs)
            y = targets.next_action.astype(jnp.int32)
            y_onehot = jax.nn.one_hot(y, A)
            
            pt = jnp.sum(y_onehot * probs, axis=-1)
            nll = -jnp.sum(y_onehot * log_probs, axis=-1)

            if use_focal_loss:
                focal_factor = jnp.power(jnp.clip(1.0 - pt, focal_eps, 1.0), focal_gamma)
                action_loss = nll * focal_factor
            else:
                action_loss = nll

            total_per_step_loss = action_loss
            
            # --- Optional SR Loss ---
            sr_loss = jnp.zeros_like(action_loss)
            if args.use_sr:
                # Target SR: [B, S, Num_States, Num_Gammas]
                log_pred_sr = jnp.log(pred_sr + 1e-8) 
                # Cross-entropy over states, sum over gammas
                sr_ce = -jnp.sum(target_sr * log_pred_sr, axis=2)
                sr_loss = jnp.sum(sr_ce, axis=-1) 
                total_per_step_loss = total_per_step_loss + (args.sr_coef * sr_loss)

            # --- Mask & Aggregate ---
            weights = targets.mask
            loss = (total_per_step_loss * weights).sum() / jnp.maximum(weights.sum(), 1.0)
            
            metrics = {
                "total_loss": loss,
                "action_loss": (action_loss * weights).sum() / jnp.maximum(weights.sum(), 1.0),
                "sr_loss": (sr_loss * weights).sum() / jnp.maximum(weights.sum(), 1.0)
            }

            return loss, metrics

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    # 4. Training Loop
    os.makedirs(args.work_dir, exist_ok=True)
    global_step = 0
    
    for epoch in range(args.epochs):
        epoch_losses = []
        start_time = time.time()
        
        for i, batch in enumerate(dataloader):
            # --- A. Data Prep ---
            obs_raw = jnp.array(batch['o_obs']) 
            # Prepare sequences (t) vs targets (t+1)
            obs_inputs = obs_raw[:, :-1]
            rew_inputs = jnp.array(batch['rew'])[:, :-1]
            
            actions = jnp.array(batch['act'])     # [B, T]
            target_action = actions[:, :-1]       # [B, T-1]
            prev_action = jnp.concatenate(
                [jnp.zeros((actions.shape[0], 1), dtype=actions.dtype), actions[:, :-2]],
                axis=1
            )                                     # [B, T-1]
            prev_reward = jnp.concatenate([jnp.zeros((actions.shape[0],1)), rew_inputs[:, :-2]], axis=1)

            
            # Prepare Masks
            mask_pad = jnp.array(batch['mask_pad'])[:, :-1]
            is_padded = 1.0 - mask_pad
            done_seq = jnp.array(batch['done'])[:, :-1]
            eff_done = jnp.maximum(done_seq, is_padded)

            inputs_jax, targets_jax = build_passive_batch_from_sequences(
                obs_seq=obs_inputs,
                prev_action_seq=prev_action,
                prev_reward_seq=prev_reward,
                next_frame_seq=None, # We are only training action prediction here
                next_other_action_seq=target_action,
                done_seq=eff_done
            )
            # Compute empirical SR targets conditionally
            target_sr = None
            if args.use_sr:
                # to convert [B, T, H, W, C] to 1D integer state IDs [B, T]
                state_indices = extract_state_indices(obs_inputs, AGENT_IDS) 
                
                # compute_empirical_sr is the lax.scan function detailed previously
                target_sr = compute_empirical_sr(
                    state_indices, 
                    eff_done, 
                    num_states=args.num_states, 
                    gammas=[0.5, 0.9, 0.99]
                )
            # --- B. Input Splitting & Formatting ---
            
            # 1. TP Input (Observer View): usually 2 channels (ID, Color)
            # Assuming obs_raw is [B, S, H, W, 2] already.
            inputs_tp = {"obs_img": inputs_jax["obs_img"]}

            # 2. FP Input (Protagonist View): usually 3 channels (ID, Color, State)
            # If dataset only has 2 channels, we pad the 3rd channel with zeros.
            obs_fp = inputs_jax["obs_img"]
            if obs_fp.shape[-1] == 2:
                B_dim, S_dim, H_dim, W_dim, _ = obs_fp.shape
                zeros = jnp.zeros((B_dim, S_dim, H_dim, W_dim, 1), dtype=obs_fp.dtype)
                obs_fp = jnp.concatenate([obs_fp, zeros], axis=-1)
            
            inputs_fp = {
                "obs_img": obs_fp,
                "obs_dir": jnp.zeros((*obs_fp.shape[:2], 4)), # Dummy dir
                "prev_action": inputs_jax["prev_action"],
                "prev_reward": inputs_jax["prev_reward"]
            }

            # 3. Hidden State Init
            # Use model to get correct shapes for FP and TP
            h_fp, h_tp = model.initialize_carry(args.batch_size)

            # --- C. Update Step ---
            state, logs = train_step(state, inputs_fp, inputs_tp, h_fp, h_tp, targets_jax, target_sr)
            
            # Logging
            loss_val = logs['total_loss'].item()
            epoch_losses.append(loss_val)
            global_step += 1
            
            if i % 10 == 0 and args.track:
                print(f"Ep {epoch} | It {i} | Loss: {loss_val:.4f}", end="\r")
                wandb.log({
                    "train/total_loss": loss_val,
                    "train/action_loss": logs['action_loss'].item(),
                    "train/sr_loss": logs['sr_loss'].item(),
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                })

        # End of Epoch
        avg_loss = np.mean(epoch_losses)
        print(f"\nEpoch {epoch} Finished | Avg Loss: {avg_loss:.4f} | Time: {time.time()-start_time:.2f}s")

        if epoch % args.save_every == 0 or epoch == args.epochs - 1:
            save_path = os.path.join(args.work_dir, f"checkpoint_{epoch}.msgpack")
            with open(save_path, "wb") as f:
                f.write(to_bytes(state))
            print(f"Saved checkpoint to: {save_path}")

    if args.track:
        wandb.finish()

if __name__ == "__main__":
    main()