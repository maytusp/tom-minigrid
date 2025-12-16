import os
import argparse
import imageio
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from flax.training import train_state
from torch.utils.data import DataLoader
# python eval_tomnet.py --data_dir ../data/trajs/MiniGrid-ToM-TwoRoomsNoSwap-13x13 --checkpoint ./checkpoints/tomnets/checkpoint_49.msgpack
# --- Imports ---
from .tom_nn import (
    AuxiliaryPredictorRNN, 
    build_passive_batch_from_sequences
)
from xminigrid.experimental.img_obs import _render_obs_tom

from .utils import (
    NpzEpisodeDataset,
    pad_collate,
)

# Scale up the video output so it isn't tiny (e.g., 56x56 pixels)
VIDEO_SCALE = 4

def load_model(config, checkpoint_path, rng):
    """
    Reconstructs the model and loads ONLY parameters from the checkpoint.
    Ignores optimizer state to prevent mismatch errors.
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # 1. Re-initialize the model structure
    model = AuxiliaryPredictorRNN(
        num_actions=config['num_actions'],
        view_size=config['fov_size'],
        predict_frame=True,
        predict_action=False,
        obs_emb_dim=config['obs_emb_dim'],
        rnn_hidden_dim=config['rnn_hidden_dim']
    )

    # 2. Create dummy inputs to initialize shapes and get the parameter structure
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
    # variables['params'] contains the empty structure we need to fill
    variables = model.init(rng, dummy_inputs, h0)
    
    # 3. Load the raw bytes
    with open(checkpoint_path, 'rb') as f:
        ckpt_contents = f.read()
    
    # 4. Deserialize to a dictionary first (Generic load)
    # This creates a raw dict: {'params': {...}, 'opt_state': {...}, 'step': ...}
    raw_state_dict = serialization.msgpack_restore(ckpt_contents)
    
    # 5. Extract only the 'params' and restore them into our model structure
    if 'params' in raw_state_dict:
        # Standard TrainState checkpoint
        loaded_params = serialization.from_state_dict(variables['params'], raw_state_dict['params'])
    else:
        # Fallback: In case the checkpoint was saved as just params
        loaded_params = serialization.from_state_dict(variables['params'], raw_state_dict)
    
    return model, loaded_params

def import_optax_dummy():
    import optax
    return optax.adam(1e-4)

def save_video(frames, path, fps=5):
    """Helper to save a list of numpy arrays as an MP4."""
    try:
        imageio.mimsave(path, frames, fps=fps, quality=8)
        print(f"Saved video to {path}")
    except Exception as e:
        print(f"Error saving video {path}: {e}")

def visualize_episode(model, params, batch, idx, output_dir):
    """
    Runs inference on a single batch item and saves MP4 videos.
    """
    # 1. Prepare Inputs for ONE episode
    obs_raw = jnp.array(batch['obs'][idx:idx+1])
    is_padded = 1.0 - jnp.array(batch['mask_pad'][idx:idx+1])
    eff_done = jnp.maximum(jnp.array(batch['done'][idx:idx+1]), is_padded)
    
    inputs_jax, _ = build_passive_batch_from_sequences(
        obs_seq=obs_raw,
        dir_seq=jnp.array(batch['dir'][idx:idx+1]),
        prev_action_seq=jnp.array(batch['act'][idx:idx+1]),
        prev_reward_seq=jnp.array(batch['rew'][idx:idx+1]),
        next_frame_seq=obs_raw[..., 0], 
        next_other_action_seq=jnp.array(batch['next_act'][idx:idx+1]),
        done_seq=eff_done
    )

    # 2. Run Inference
    h0 = model.initialize_carry(batch_size=1)
    outputs, _ = model.apply({'params': params}, inputs_jax, h0)
    
    # [1, T, H, W, V_vocab] -> [1, T, H, W]
    pred_ids = jnp.argmax(outputs['frame_logits'], axis=-1) 

    # 3. Render Loop
    frames_gt = []
    frames_comp = []
    
    T_valid = int(batch['length'])
    
    # Base filename without extension
    base_name = batch['file_ids'][idx].replace('.npz', '')
    print(f"Rendering episode {base_name} (Length {T_valid})...")
    
    for t in range(T_valid):
        if t + 1 >= T_valid:
            break
            
        # --- A. Extract Data ---
        gt_next_frame = obs_raw[0, t+1] 
        pred_frame = pred_ids[0, t] 
        print(f"pred_ids {pred_ids.shape}")
        # print("GT: \n", gt_next_frame[:,:,0])
        # print("predicted: \n", pred_frame)
        # Borrow GT colors for visualization
        gt_color_id = gt_next_frame[:, :, 1]
        pred_frame = jnp.stack([pred_frame, gt_color_id], axis=-1)
        
        # --- B. Render Pixels ---
        gt_pixels = _render_obs_tom(gt_next_frame) 
        pred_pixels = _render_obs_tom(pred_frame) 
        
        gt_img = np.array(gt_pixels, dtype=np.uint8)
        pred_img = np.array(pred_pixels, dtype=np.uint8)
        
        # --- C. Upscale (Make it watchable on standard players) ---
        # Repeat pixels along H and W dimensions
        gt_img = gt_img.repeat(VIDEO_SCALE, axis=0).repeat(VIDEO_SCALE, axis=1)
        pred_img = pred_img.repeat(VIDEO_SCALE, axis=0).repeat(VIDEO_SCALE, axis=1)
        
        # --- D. Collect Frames ---
        # 1. Comparison Frame (Left: GT, Right: Prediction)
        combined_img = np.concatenate([gt_img, pred_img], axis=1)
        
        frames_gt.append(gt_img)
        frames_comp.append(combined_img)

    # 4. Save MP4s
    path_gt = os.path.join(output_dir, f"{base_name}_gt.mp4")
    path_comp = os.path.join(output_dir, f"{base_name}_comp.mp4")
    
    save_video(frames_gt, path_gt, fps=5)
    save_video(frames_comp, path_comp, fps=5)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./logs/trajs/MiniGrid-Protagonist-ProcGen-9x9vs9")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/observers/static/checkpoint_49.msgpack")
    parser.add_argument("--output_dir", type=str, default="./logs/observer_eval/static")
    
    # Model Args (Must match training)
    parser.add_argument("--fov_size", type=int, default=9)
    parser.add_argument("--num_actions", type=int, default=6) 
    parser.add_argument("--obs_emb_dim", type=int, default=16)
    parser.add_argument("--rnn_hidden_dim", type=int, default=256)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Setup Data
    dataset = NpzEpisodeDataset(args.data_dir, max_files=10)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=pad_collate)
    
    # 2. Setup Model
    rng = jax.random.key(0)
    config = vars(args)
    model, params = load_model(config, args.checkpoint, rng)
    
    # 3. Visualize
    for i, batch in enumerate(dataloader):
        visualize_episode(model, params, batch, 0, args.output_dir)

if __name__ == "__main__":
    main()