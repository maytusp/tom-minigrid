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
import functools
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./logs/train_trajs/tworoom_noswap_with_sr")
    parser.add_argument("--work_dir", type=str, default="./checkpoints/observers/tworoom-noswap/tp-sr/")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    
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
    parser.add_argument("--sr_coef", type=float, default=0.1, 
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
    def train_step(state, inputs_fp, inputs_tp, h_fp, h_tp, targets: PassiveTargets):
        def loss_fn(params):
            logits, pred_sr, _, _ = state.apply_fn({'params': params}, inputs_fp, h_fp, inputs_tp, h_tp)
            A = logits.shape[-1]
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            probs = jnp.exp(log_probs)
            y_onehot = jax.nn.one_hot(targets.next_action.astype(jnp.int32), A)
            
            pt = jnp.sum(y_onehot * probs, axis=-1)
            action_loss = -jnp.sum(y_onehot * log_probs, axis=-1) * jnp.power(jnp.clip(1.0 - pt, 1e-8, 1.0), 3.0)
            
            if args.use_sr and targets.target_sr is not None:
                log_pred_sr = jnp.log(pred_sr + 1e-8) 
                sr_loss = jnp.sum(-jnp.sum(targets.target_sr * log_pred_sr, axis=2), axis=-1) 
            else:
                sr_loss = jnp.zeros_like(action_loss)
            
            total_loss = action_loss + (args.sr_coef * sr_loss)
            loss = (total_loss * targets.mask).sum() / jnp.maximum(targets.mask.sum(), 1.0)
            
            metrics = {
                "total_loss": loss,
                "action_loss": (action_loss * targets.mask).sum() / jnp.maximum(targets.mask.sum(), 1.0),
                "sr_loss": (sr_loss * targets.mask).sum() / jnp.maximum(targets.mask.sum(), 1.0)
            }
            return loss, metrics

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(state.params)
        return state.apply_gradients(grads=grads), metrics

    # 4. Training Loop
    os.makedirs(args.work_dir, exist_ok=True)
    global_step = 0
    
    for epoch in range(args.epochs):
        epoch_losses = []
        start_time = time.time()
        
        for i, batch in enumerate(dataloader):
            obs_raw = jnp.array(batch['o_obs']) # [B, T]
            obs_inputs = obs_raw
            rew_inputs = jnp.array(batch['rew'])
            actions = jnp.array(batch['act'])     # [B, T]
            target_action = actions       # [B, T]
            prev_action = jnp.concatenate(
                [jnp.zeros((actions.shape[0], 1), dtype=actions.dtype), actions[:, :-1]],
                axis=1
            )                                     # [B, T]
            prev_reward = jnp.concatenate([jnp.zeros((actions.shape[0],1)), rew_inputs[:, :-1]], axis=1)

            
            # Prepare Masks
            mask_pad = jnp.array(batch['mask_pad'])
            is_padded = 1.0 - mask_pad
            done_seq = jnp.array(batch['done'])
            eff_done = jnp.maximum(done_seq, is_padded)

            target_sr_batch = None
            if args.use_sr:
                sr_array = np.stack(batch['target_sr']) if isinstance(batch['target_sr'], list) else batch['target_sr']
                target_sr_batch = jnp.array(sr_array)

            inputs_jax, targets_jax = build_passive_batch_from_sequences(
                obs_seq=obs_inputs,
                prev_action_seq=prev_action,
                prev_reward_seq=prev_reward,
                next_frame_seq=None, 
                next_other_action_seq=target_action,
                done_seq=eff_done,
                target_sr_seq=target_sr_batch,
            )
            
            inputs_tp = {"obs_img": inputs_jax["obs_img"]}

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

            # Use model to get correct shapes for FP and TP
            h_fp, h_tp = model.initialize_carry(args.batch_size)

            state, logs = train_step(state, inputs_fp, inputs_tp, h_fp, h_tp, targets_jax)

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