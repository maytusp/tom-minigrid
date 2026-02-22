import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax.serialization import from_bytes
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Import your modules (Keep these consistent with your project structure)
from .tom_nn import (
    create_model,
    build_passive_batch_from_sequences
)
from .utils import (
    NpzEpisodeDataset,
    pad_collate
)

class TrainState(train_state.TrainState):
    pass

def load_checkpoint(state, checkpoint_path):
    with open(checkpoint_path, "rb") as f:
        state = from_bytes(state, f.read())
    return state

def plot_confusion_matrix(y_true, y_pred, action_labels=None, save_path=None):
    """
    Generates and saves a confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize by row (true class) to see recall/sensitivity
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) # Handle divide by zero for classes not present
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=action_labels if action_labels else "auto",
                yticklabels=action_labels if action_labels else "auto")
    
    plt.ylabel('True Action')
    plt.xlabel('Predicted Action')
    plt.title('Action Prediction Confusion Matrix (Normalized)')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    # Path args
    parser.add_argument("--data_dir", type=str, default="./logs/train_trajs/tworoom_noswap")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/observers/tworoom-noswap/tp/checkpoint_49.msgpack", help="Path to the .msgpack file to evaluate")
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    
    # Config Args (Must match training!)
    parser.add_argument("--model_type", type=str, default="third_person", choices=["third_person", "dual_perspective"])
    parser.add_argument("--num_actions", type=int, default=6) 
    parser.add_argument("--fp_emb", type=int, default=16)
    parser.add_argument("--fp_rnn", type=int, default=256)
    parser.add_argument("--tp_emb", type=int, default=16)
    parser.add_argument("--tp_rnn", type=int, default=256)
    
    # Evaluation specific
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sample_trajs", type=int, default=1000, help="Number of trajectories to evaluate")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Setup Data (Same as train)
    dataset = NpzEpisodeDataset(args.data_dir)
    # We use shuffle=False to get consistent evaluation samples
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, 
        collate_fn=pad_collate, num_workers=0, drop_last=False
    )

    # 2. Re-create Model Structure
    rng = jax.random.key(args.seed)
    config = vars(args)
    # Note: We don't need the protagonist checkpoint here as we are loading the full observer state
    config['p_checkpoint'] = None 
    
    model, params = create_model(args.model_type, config, rng)
    
    # Initialize dummy state to structure the loading
    tx = optax.adam(1e-4) # Learning rate doesn't matter for eval
    state = TrainState.create(apply_fn=model.apply, params=params['params'], tx=tx)
    
    # 3. Load Weights
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    state = load_checkpoint(state, args.checkpoint_path)

    # 4. Evaluation Loop
    print(f"Evaluating on {args.sample_trajs} trajectories...")
    
    all_preds = []
    all_targets = []
    trajs_processed = 0
    
    # JIT the inference step for speed
    @jax.jit
    def eval_step(params, inputs_fp, inputs_tp, h_fp, h_tp):
        logits, _, _ = state.apply_fn(
            {'params': params},
            inputs_fp, h_fp, inputs_tp, h_tp
        )
        return logits

    for batch in dataloader:
        if trajs_processed >= args.sample_trajs:
            break

        # --- Data Prep (Copied from Train Code) ---
        obs_raw = jnp.array(batch['o_obs']) 
        obs_inputs = obs_raw[:, :-1]
        rew_inputs = jnp.array(batch['rew'])[:, :-1]
        
        actions = jnp.array(batch['act'])     
        target_action = actions[:, :-1]       
        prev_action = jnp.concatenate(
            [jnp.zeros((actions.shape[0], 1), dtype=actions.dtype), actions[:, :-2]],
            axis=1
        )                                     
        prev_reward = jnp.concatenate([jnp.zeros((actions.shape[0],1)), rew_inputs[:, :-2]], axis=1)

        mask_pad = jnp.array(batch['mask_pad'])[:, :-1]
        is_padded = 1.0 - mask_pad
        done_seq = jnp.array(batch['done'])[:, :-1]
        eff_done = jnp.maximum(done_seq, is_padded)

        inputs_jax, targets_jax = build_passive_batch_from_sequences(
            obs_seq=obs_inputs,
            prev_action_seq=prev_action,
            prev_reward_seq=prev_reward,
            next_frame_seq=None, 
            next_other_action_seq=target_action,
            done_seq=eff_done
        )

        # --- Input Formatting ---
        inputs_tp = {"obs_img": inputs_jax["obs_img"]}
        obs_fp = inputs_jax["obs_img"]
        if obs_fp.shape[-1] == 2:
            B_dim, S_dim, H_dim, W_dim, _ = obs_fp.shape
            zeros = jnp.zeros((B_dim, S_dim, H_dim, W_dim, 1), dtype=obs_fp.dtype)
            obs_fp = jnp.concatenate([obs_fp, zeros], axis=-1)
        
        inputs_fp = {
            "obs_img": obs_fp,
            "obs_dir": jnp.zeros((*obs_fp.shape[:2], 4)), 
            "prev_action": inputs_jax["prev_action"],
            "prev_reward": inputs_jax["prev_reward"]
        }

        # --- Inference ---
        h_fp, h_tp = model.initialize_carry(obs_raw.shape[0])
        logits = eval_step(state.params, inputs_fp, inputs_tp, h_fp, h_tp)
        
        # --- Metrics Extraction ---
        preds = jnp.argmax(logits, axis=-1)  # [B, S]
        targets = targets_jax.next_action    # [B, S]
        mask = targets_jax.mask              # [B, S]

        # Flatten and filter by mask
        preds_flat = preds.flatten()
        targets_flat = targets.flatten()
        mask_flat = mask.flatten()

        valid_indices = mask_flat > 0
        
        batch_preds = preds_flat[valid_indices]
        batch_targets = targets_flat[valid_indices]

        all_preds.extend(np.array(batch_preds))
        all_targets.extend(np.array(batch_targets))
        
        trajs_processed += args.batch_size
        print(f"Processed {min(trajs_processed, args.sample_trajs)} / {args.sample_trajs} trajectories...", end="\r")

    print("\nProcessing complete.")

    # 5. Visualization & Reporting
    
    # Define Action Labels (Modify based on your environment)
    action_map = {
        0: "Forward", 1: "Left", 2: "Right", 3: "Open/Close", 
        4: "", 5: "", 6: ""
    }
    # Create labels list based on what actually appears in data or num_actions
    labels = [action_map.get(i, str(i)) for i in range(args.num_actions)]

    print("\n--- Classification Report ---")
    print(classification_report(all_targets, all_preds, target_names=labels[:len(np.unique(all_targets))]))

    print("Generating Confusion Matrix...")
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(all_targets, all_preds, action_labels=labels, save_path=cm_path)

if __name__ == "__main__":
    main()