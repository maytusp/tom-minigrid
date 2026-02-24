import os
import glob
import argparse
import numpy as np
from tqdm import tqdm

def compute_sr_numpy(obs_seq, done_seq, agent_ids, num_states, gammas):
    """
    Computes the normalized empirical Successor Representation targets.
    obs_seq: [T, H, W, C]
    done_seq: [T]
    """
    T = len(done_seq)
    
    # 1. Find agent state indices
    id_channel = obs_seq[..., 0]
    is_agent = np.isin(id_channel, agent_ids)
    
    # Flatten spatial dims to find the 1D location
    flat_is_agent = is_agent.reshape(T, -1)
    
    # Check if agent actually exists in the frame (handles padding)
    has_agent = np.any(flat_is_agent, axis=-1) 
    state_indices = np.argmax(flat_is_agent, axis=-1)
    
    # 2. Compute one-hot states [T, Num_States]
    one_hot_states = np.eye(num_states, dtype=np.float32)[state_indices]
    # Zero out states where the agent isn't even in the frame
    one_hot_states = one_hot_states * has_agent[:, None] 
    
    # 3. Iterate backwards to compute discounted future occupancy
    sr_unnorm = np.zeros((T, num_states, len(gammas)), dtype=np.float32)
    current_occupancy = np.zeros((num_states, len(gammas)), dtype=np.float32)
    gammas_arr = np.array(gammas, dtype=np.float32)
    
    for t in reversed(range(T)):
        # If episode ends, future occupancy is reset (can't cross episode boundaries)
        if done_seq[t]:
            current_occupancy = np.zeros_like(current_occupancy)
            
        # State at time t (shape: [Num_States, 1])
        s_t = np.expand_dims(one_hot_states[t], axis=-1) 
        
        # SR_t = S_t + gamma * SR_{t+1}
        current_occupancy = s_t + gammas_arr * current_occupancy
        sr_unnorm[t] = current_occupancy
        
    # 4. Normalize so the sum over states is 1 for each gamma
    Z = np.sum(sr_unnorm, axis=1, keepdims=True)
    target_sr = sr_unnorm / np.maximum(Z, 1e-8)
    
    return target_sr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./logs/val_trajs/tworoom_noswap", help="Path to original data")
    parser.add_argument("--output_dir", type=str, default="./logs/val_trajs/tworoom_noswap_with_sr", help="Path to save processed data")
    parser.add_argument("--agent_ids", type=int, nargs="+", default=[13, 14, 15, 16])
    parser.add_argument("--num_states", type=int, default=81)
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.5, 0.9, 0.99])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(args.input_dir, "*.npz"))
    print(f"Found {len(files)} files to process. Starting...")
    for filepath in tqdm(files):
        filename = os.path.basename(filepath)
        out_path = os.path.join(args.output_dir, filename)
        
        if os.path.exists(out_path):
            continue
            
        with np.load(filepath, allow_pickle=True) as data:
            new_data = {key: data[key] for key in data.files}
            
            # 1. Get the observation sequence
            # Use 'o_sym' since that is the raw key in your dataset
            obs_seq = new_data['o_sym'] 
            T = obs_seq.shape[0]
            
            # 2. Reconstruct the done_seq
            # Since each file is one episode, 'done' is usually 0 everywhere and 1 at the very end.
            done_seq = np.zeros(T, dtype=np.float32)
            
            # (Optional) If your 'meta' dictionary contains a specific 'done' array, extract it:
            if 'meta' in new_data and isinstance(new_data['meta'].item(), dict) and 'done' in new_data['meta'].item():
                done_seq = np.array(new_data['meta'].item()['done'], dtype=np.float32)
            else:
                done_seq[-1] = 1.0  # Default: episode finishes at the last frame
            
            # 3. Compute SR
            target_sr = compute_sr_numpy(
                obs_seq=obs_seq,
                done_seq=done_seq,
                agent_ids=args.agent_ids,
                num_states=args.num_states,
                gammas=args.gammas
            )
            
            new_data['target_sr'] = target_sr
            
            np.savez_compressed(out_path, **new_data)
            
    print(f"Finished! Processed dataset saved to: {args.output_dir}")

if __name__ == "__main__":
    main()