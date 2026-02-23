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

class TrainState(train_state.TrainState):
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./logs/train_trajs/tworoom_noswap")
    parser.add_argument("--work_dir", type=str, default="./checkpoints/observers/tworoom-noswap/tp/")
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

    # 3. Define Train Step
    @jax.jit
    def train_step(state, inputs_fp, inputs_tp, h_fp, h_tp, targets: PassiveTargets):
        print("Compiling train_step... (This should only print ONCE!)")
        def loss_fn(params, *, use_focal_loss: bool = True, focal_gamma: float = 5.0, focal_eps: float = 1e-8):
            logits, _, _, _ = state.apply_fn(
                {'params': params},
                inputs_fp, h_fp, inputs_tp, h_tp
            )  # logits: [B, S, A]

            A = logits.shape[-1]

            # log_probs + probs
            log_probs = jax.nn.log_softmax(logits, axis=-1)     # [B,S,A]
            probs = jnp.exp(log_probs)                          # [B,S,A]

            # true-class log prob and prob
            y = targets.next_action.astype(jnp.int32)           # [B,S]
            y_onehot = jax.nn.one_hot(y, A)                     # [B,S,A]

            log_pt = jnp.sum(y_onehot * log_probs, axis=-1)     # [B,S]
            pt = jnp.sum(y_onehot * probs, axis=-1)             # [B,S]

            nll = -log_pt                                       # [B,S]

            # --- Focal factor ---
            if use_focal_loss:
                # (1 - pt)^gamma ; clamp for numerical safety
                focal_factor = jnp.power(jnp.clip(1.0 - pt, focal_eps, 1.0), focal_gamma)  # [B,S]
                per_step_loss = nll * focal_factor
            else:
                per_step_loss = nll

            # --- Mask + weights ---
            weights = targets.mask                     # [B,S]
            loss = (per_step_loss * weights).sum() / jnp.maximum(weights.sum(), 1.0)

            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, {"total_loss": loss}

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
            state, logs = train_step(state, inputs_fp, inputs_tp, h_fp, h_tp, targets_jax)
            
            # Logging
            loss_val = logs['total_loss'].item()
            epoch_losses.append(loss_val)
            global_step += 1
            
            if i % 10 == 0 and args.track:
                print(f"Ep {epoch} | It {i} | Loss: {loss_val:.4f}", end="\r")
                wandb.log({
                    "train/total_loss": loss_val,
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