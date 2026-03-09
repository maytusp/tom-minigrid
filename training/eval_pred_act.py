import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.serialization import msgpack_restore, from_state_dict, from_bytes
from flax.core import freeze

import xminigrid
from xminigrid.wrappers import AllocentricObservationWrapper, GymAutoResetWrapper
from xminigrid.experimental.img_obs import RGBImgObservationWrapper
from xminigrid.experimental.render_from_symbolic import _render
import imageio.v2 as imageio

# Import P's network
from .nn import ActorCriticRNN
# Import O's networks (Standard + Dual) and factory
from .tom_nn import create_model, DualPerspectivePredictor, ThirdPersonPredictor
from .utils import _dir_to_id

def get_door_sequence(obs_seq):
    """
    Extracts coordinates of doors that transitioned from CLOSED to OPEN.
    obs_seq shape: (T, 9, 9, 2)
    """
    T = obs_seq.shape[0]
    if T < 2:
        return []

    star_reached = False
    visited_sequence = []
    # Use a set to ensure we don't log the same door-opening event 
    # multiple times if the logic triggers over consecutive frames
    opened_doors = set()

    for t in range(1, T):
        # 1. Identify if the star is reached (Star ID = 12)
        # We only care about doors opened AFTER the star is gone
        if not star_reached:
            star_present = np.any(obs_seq[t, ..., 0] == 12)
            if not star_present:
                star_reached = True
            continue # Skip processing until star is reached

        # 2. Compare current grid with previous grid (Channel 0 only)
        prev_grid = obs_seq[t-1, ..., 0]
        curr_grid = obs_seq[t, ..., 0]

        # 3. Find coordinates where state changed from 9 (CLOSED) to 10 (OPEN)
        # This captures the "Toggle" effect regardless of agent position
        changed_to_open = (prev_grid == 9) & (curr_grid == 10)
        coords = np.argwhere(changed_to_open)

        for c in coords:
            door_pos = tuple(c)
            if door_pos not in opened_doors:
                visited_sequence.append(door_pos)
                opened_doors.add(door_pos)

    return visited_sequence

# --- Configs ---
@dataclass
class PModelCfg:
    """Config for Protagonist (P)"""
    obs_emb_dim: int = 16
    action_emb_dim: int = 16
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1
    head_hidden_dim: int = 128
    img_obs: bool = False
    enable_bf16: bool = False
    use_color: bool = True 

@dataclass
class OModelCfg:
    """Config for Observer (O)"""
    obs_emb_dim: int = 16
    rnn_hidden_dim: int = 256
    fov_size: int = 9
    num_actions: int = 6 

# --- Setup & Loading ---

def build_env(env_id: str, img_obs: bool):
    env, env_params = xminigrid.make(env_id)
    env = GymAutoResetWrapper(env)
    env = AllocentricObservationWrapper(env)
    if img_obs:
        env = RGBImgObservationWrapper(env)
    return env, env_params

def load_p_params(checkpoint_path: str, net, env, env_params):
    """Load Protagonist (P) parameters from msgpack."""
    shapes = env.observation_shape(env_params)
    init_obs = {
        "obs_img": jnp.zeros((1, 1, *shapes["p_img"])),
        "obs_dir": jnp.zeros((1, 1, shapes["direction"])),
        "prev_action": jnp.zeros((1, 1), dtype=jnp.int32),
        "prev_reward": jnp.zeros((1, 1)),
    }
    init_hstate = net.initialize_carry(batch_size=1)
    rng = jax.random.key(0)
    target_vars = net.init(rng, init_obs, init_hstate)
    
    with open(checkpoint_path, "rb") as f:
        raw = msgpack_restore(f.read())
    
    raw_params = raw["params"] if (isinstance(raw, dict) and "params" in raw) else raw
    # We use 'target_vars["params"]' as the template structure
    loaded_params = from_state_dict(target_vars["params"], raw_params)
    
    print(f"[P-Loader] Loaded from {checkpoint_path}")
    return freeze({"params": loaded_params})

import jax
import jax.numpy as jnp
from flax.serialization import msgpack_restore, from_state_dict
from flax.core import freeze

def count_params(params_tree):
    """Utility to count total parameters in a PyTree."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params_tree))

def load_o_params(checkpoint_path, net, cfg, model_type="third_person"):
    """
    Load Observer (O) Parameters. 
    Handles both standard and dual models by using the network's own init structure.
    """
    rng = jax.random.key(0)
    
    # Dummy inputs for initialization
    dummy_fp = {
        "obs_img": jnp.zeros((1, 1, 9, 9, 3), dtype=jnp.int32), 
        "obs_dir": jnp.zeros((1, 1, 4)),
        "prev_action": jnp.zeros((1, 1), dtype=jnp.int32),
        "prev_reward": jnp.zeros((1, 1))
    }
    # Note: assuming cfg.fov_size is available
    dummy_tp = {"obs_img": jnp.zeros((1, 1, cfg.fov_size, cfg.fov_size, 2), dtype=jnp.int32)}
    
    # Dual/TP models usually return 2 hidden states
    h_fp, h_tp = net.initialize_carry(batch_size=1)
    
    # Initialize to get correct structure
    variables = net.init(rng, dummy_fp, h_fp, dummy_tp, h_tp)
    target_params = variables['params']
    
    # Load raw bytes -> msgpack dict
    with open(checkpoint_path, "rb") as f:
        try:
            raw_data = msgpack_restore(f.read())
            if isinstance(raw_data, dict) and 'params' in raw_data:
                raw_params = raw_data['params']
            else:
                raw_params = raw_data
        except:
            f.seek(0)
            raw_params = msgpack_restore(f.read())['params']

    # --- NEW: Parameter Counting and Comparison ---
    expected_count = count_params(target_params)
    checkpoint_count = count_params(raw_params)
    
    if expected_count != checkpoint_count:
        print("\n[O-Loader] ⚠️ WARNING: Architecture Mismatch Detected!")
        print(f"  -> Current Model requires: {expected_count:,} parameters")
        print(f"  -> Checkpoint provides:    {checkpoint_count:,} parameters")
        print("  -> Unmatched model layers will remain randomly initialized.\n")
    else:
        print(f"\n[O-Loader] Architecture matches perfectly ({expected_count:,} params).")

    # Robust restoration (maps raw_params onto target_params structure)
    loaded_params = from_state_dict(target_params, raw_params)
    
    print(f"[O-Loader] Successfully mapped parameters from {checkpoint_path}")
    return freeze(loaded_params)

# --- Dual Rollout Logic ---

def run_dual_rollout(
    env,
    env_params,
    net_p, params_p,
    net_o, params_o,
    *,
    episodes: int = 1,
    max_steps: int = 80,
    seed: int = 0,
    out_dir: str = "trajs_pred",
    observer_r: int = 8,
    observer_c: int = 5,
    fov_dir: str = "up",
    fov_size: int = 9,
    model_type: str = "third_person",
    
):
    os.makedirs(out_dir, exist_ok=True)
    
    # P Init
    h0_p = net_p.initialize_carry(batch_size=1)
    
    # O Init (Dual model returns 2 states: h_fp, h_tp)
    if model_type == "transformed_fp":
        h0_o_fp = net_o.initialize_carry(batch_size=1)
        h0_o_tp = None
    else:
        h0_o_fp, h0_o_tp = net_o.initialize_carry(batch_size=1)
    
    dir_id = _dir_to_id(fov_dir)

    @jax.jit
    def run_one_episode(rng):
        rng, rng_reset = jax.random.split(rng)
        timestep_act = env.reset(env_params, rng_reset)
        timestep_pred = jax.tree_util.tree_map(lambda x: x, timestep_act)

        prev_action_act = jnp.asarray(0, dtype=jnp.int32)
        prev_action_pred = jnp.asarray(0, dtype=jnp.int32)

        init_carry = (
            timestep_act, timestep_pred, 
            h0_p,           # P's hidden state
            h0_o_fp, h0_o_tp, # O's hidden states (FP, TP)
            prev_action_act, prev_action_pred, 
            rng
        )

        def step_fn(carry, _):
            ts_act, ts_pred, h_p, h_o_fp, h_o_tp, pa_act, pa_pred, _rng = carry
            
            # --- 1. Observations ---
            obs_p = ts_act.observation["p_img"]
            obs_o = ts_pred.observation["o_img"]
            # obs_o = crop_fov_symbolic_allocentric(
            #     grid_sym=obs_o, 
            #     r=observer_r, 
            #     c=observer_c, 
            #     view_size=fov_size, 
            #     dir_id=dir_id
            # )
            # Protagonist Inference
            in_p = {
                "obs_img": obs_p[None, None, ...],
                "prev_action": pa_act[None, None],
                "prev_reward": ts_act.reward[None, None],
                "obs_dir": jnp.zeros((1, 1, 4))
            }
            
            _rng, rng_p, rng_o_sample = jax.random.split(_rng, 3)
            dist_p, _, new_h_p = net_p.apply(params_p, in_p, h_p)
            action_p = dist_p.sample(seed=rng_p).squeeze()
            # action_p = jnp.argmax(dist_p, axis=-1).squeeze()
            # Observer Inference
            in_o_fp = {
                "obs_img": obs_o[None, None, ...],
                "obs_dir": jnp.zeros((1, 1, 4)),
                "prev_action": pa_pred[None, None],
                "prev_reward": ts_pred.reward[None, None],
            }

            in_o_tp = {
                "obs_img": obs_o[None, None, ...],
            }
            if model_type == "transformed_fp":
                # If we use the Transformed FP model, the structure of network is the same as net_p
                # So the output is distribution object, not logits (distribution parameters) 
                dist_o, _, new_h_o_fp, _ = net_o.apply(params_o,
                                                in_o_fp, h_o_fp)
                _rng, rng_o = jax.random.split(_rng, 2)
                action_o = dist_o.sample(seed=rng_o).squeeze()
                new_h_o_tp = None # No TP state for this model
            else:
                logits_o, _, new_h_o_fp, new_h_o_tp = net_o.apply(
                    {"params": params_o}, 
                    in_o_fp, h_o_fp, in_o_tp, h_o_tp
                )
                probs_o = jax.nn.softmax(logits_o)
                entropy = -jnp.sum(probs_o * jnp.log(probs_o + 1e-6))
                action_o = jnp.argmax(logits_o, axis=-1).squeeze()

            # Check if we are in the "Predicted" phase (Star is gone)
            star_visible = jnp.any(obs_o[..., 0] == 12)
            
            action_for_pred = jnp.where(star_visible, action_p, action_o)
            # --- 5. Environment Step ---
            new_ts_act = env.step(env_params, ts_act, action_p)
            new_ts_pred = env.step(env_params, ts_pred, action_for_pred)

            new_carry = (
                new_ts_act, new_ts_pred, 
                new_h_p, 
                new_h_o_fp, new_h_o_tp,
                action_p, action_for_pred, 
                _rng
            )
            act_pos = ts_act.state.agent.position
            pred_pos = ts_pred.state.agent.position

            outs = {
                "act_o_img": ts_act.observation["o_img"],
                "pred_o_img": ts_pred.observation["o_img"],
                "act_pos": act_pos,        
                "pred_pos": pred_pos,           
                "action_p": action_p,
                "action_o": action_o,
                "used_action": action_for_pred,
                "star_visible": star_visible,
                "reward_act": ts_act.reward,
                "done_act": ts_act.last(),
                "done_pred": ts_pred.last(),
            }
            return new_carry, outs


        final_carry, scan_out = jax.lax.scan(step_fn, init_carry, None, length=max_steps)
        return scan_out


    def render_traj(scan_out, T_act, T_pred):
        # 1. Determine the maximum length for the video
        T_max = max(T_act, T_pred)

        # 2. Slice the raw symbolic grids up to their respective 'done' points
        #    We only process valid frames to save compute
        o_img_act = scan_out["act_o_img"][:T_act]
        o_img_pred = scan_out["pred_o_img"][:T_pred]
        
        # 3. Render and Crop (JAX operations)
        #    Note: This returns arrays of shape (T_act, H, W, C) and (T_pred, H, W, C)
        rgb_act = jax.vmap(_render)(o_img_act)
        rgb_pred = jax.vmap(_render)(o_img_pred)

        # def _crop(rgb, sym):
        #     Hc, Wc = sym.shape[0], sym.shape[1]
        #     return crop_fov_from_allocentric_rgb(rgb, Hc, Wc, observer_r, observer_c, fov_size, dir_id)
        
        # rgb_act_crop = rgb_act # jax.vmap(_crop)(rgb_act, o_img_act)
        # rgb_pred_crop = rgb_pred # jax.vmap(_crop)(rgb_pred, o_img_pred)
        
        # Convert to Numpy for easy padding
        vid_act = np.array(rgb_act)
        vid_pred = np.array(rgb_pred)
        
        # 4. Pad with Black Frames (Zeros)
        #    Shape is (Time, Height, Width, Channels)
        H, W, C = vid_act.shape[1:]
        
        # Pad Act if it finished early
        if T_act < T_max:
            pad_len = T_max - T_act
            black_frames = np.zeros((pad_len, H, W, C), dtype=vid_act.dtype)
            vid_act = np.concatenate([vid_act, black_frames], axis=0)
            
        # Pad Pred if it finished early
        if T_pred < T_max:
            pad_len = T_max - T_pred
            black_frames = np.zeros((pad_len, H, W, C), dtype=vid_pred.dtype)
            vid_pred = np.concatenate([vid_pred, black_frames], axis=0)
            
        # 5. Combine Side-by-Side
        #    Create a white separator line
        sep = np.ones((T_max, H, 2, C), dtype=vid_act.dtype) * 255
        
        combined = np.concatenate([vid_act, sep, vid_pred], axis=2)
        return combined

    rng = jax.random.key(seed)
    
    correct_predictions = 0
    first_door_correct = 0
    failed_predictions = 0
    ignored_episodes = 0
    valid_episodes = 0

    for ep in range(episodes):
        rng, sub_rng = jax.random.split(rng)
        out = run_one_episode(sub_rng)
        
        dones_act = np.array(out["done_act"])
        dones_pred = np.array(out["done_pred"])

        if dones_act.any():
            T_act = np.argmax(dones_act) + 1  # +1 to include the step where done=True
        else:
            T_act = max_steps 

        if dones_pred.any():
            T_pred = np.argmax(dones_pred) + 1
        else:
            T_pred = max_steps
      
        reward = np.sum(out["reward_act"][:T_act])

        # Extract the grids and positions up to time T
        act_grids = np.array(out["act_o_img"][:T_act])
        pred_grids = np.array(out["pred_o_img"][:T_pred])
        act_positions = np.array(out["act_pos"][:T_act])
        pred_positions = np.array(out["pred_pos"][:T_pred])
        
        # Get sequences of opened doors
        doors_act = get_door_sequence(act_grids)
        doors_pred = get_door_sequence(pred_grids)

        # Variables to track specific failures
        is_full_match = False
        is_first_match = False

        if len(doors_act) == 0:
            alignment_status = "IGNORED (Actual agent opened no doors)"
            ignored_episodes += 1
        else:
            # --- Check First Door Alignment ---
            # Did the observer at least get the first door right?
            if len(doors_pred) > 0 and doors_act[0] == doors_pred[0]:
                is_first_match = True
                first_door_correct += 1  # Ensure this counter is initialized outside loop
            
            # --- Check Full Sequence Alignment ---
            if doors_act == doors_pred:
                is_full_match = True
                correct_predictions += 1
                alignment_status = "CORRECT (Full Match)"
            else:
                # Descriptive status for logging
                num_match = sum(1 for a, p in zip(doors_act, doors_pred) if a == p)
                alignment_status = (
                    f"MISMATCH | Act: {len(doors_act)}, Pred: {len(doors_pred)}. "
                    f"First Door Match: {is_first_match}"
                )
                failed_predictions += 1

        print(f"Episode {ep:03d}: {alignment_status}")
        print(f"  > Actual: {doors_act}")
        print(f"  > Pred:   {doors_pred}")

        video_frames = render_traj(out, T_act, T_pred)
        video_np = np.array(video_frames)
        
        vid_path = os.path.join(out_dir, f"ep_{ep:03d}_pred_comparison.mp4")
        imageio.mimsave(vid_path, video_np, fps=5)
        
        np.savez(
            os.path.join(out_dir, f"ep_{ep:03d}_data.npz"),
            action_p=out["action_p"][:T_act],
            action_o=out["action_o"][:T_pred],
        )
        print(f"Saved {vid_path}")

    valid_episodes = episodes - ignored_episodes
    if valid_episodes > 0:
        print(f"\n--- Results over {valid_episodes} valid episodes ---")
        print(f"Full Sequence Accuracy: {correct_predictions / valid_episodes:.2%}")
        print(f"First Door Accuracy:    {first_door_correct / valid_episodes:.2%}")
    else:
        print("No valid episodes found (Protag never opened a door).")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p_checkpoint", type=str, default="./checkpoints/MiniGrid-ToM-TwoRoomsNoSwap-9x9vs9/MiniGrid-ToM-TwoRoomsNoSwap-9x9vs9-ppo_final.msgpack")
    parser.add_argument("--checkpoint", type=str, default="", help="Observer checkpoint path")
    parser.add_argument("--model_type", type=str, default="transformed_fp", choices=["third_person", "dual_perspective", "transformed_fp"])
    parser.add_argument("--use_sr", action="store_true", default=False, 
                            help="Enable Successor Representation prediction and loss.")
    parser.add_argument("--num_states", type=int, default=81, 
                        help="Total number of discrete states for the SR (update based on your grid size).")
    parser.add_argument("--env_id", type=str, default="MiniGrid-ToM-TwoRoomsSwap-9x9vs9-d4")
    parser.add_argument("--vid_out_dir", type=str, default="logs/eval_pred_action/FPNet-RL/swap_delay4")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    # O args (must match training)
    parser.add_argument("--o_fov", type=int, default=9)
    parser.add_argument("--num_actions", type=int, default=6) 
    parser.add_argument("--fp_emb", type=int, default=16)
    parser.add_argument("--fp_rnn", type=int, default=256)
    parser.add_argument("--tp_emb", type=int, default=16)
    parser.add_argument("--tp_rnn", type=int, default=256)
    args = parser.parse_args()


    print("eval checkpoint", args.checkpoint)
    
    # 1. Env
    env, env_params = build_env(args.env_id, img_obs=False) 
    
    # 2. Protagonist (P) Setup
    p_cfg = PModelCfg()
    net_p = ActorCriticRNN(
        num_actions=env.num_actions(env_params),
        obs_emb_dim=p_cfg.obs_emb_dim,
        action_emb_dim=p_cfg.action_emb_dim,
        rnn_hidden_dim=p_cfg.rnn_hidden_dim,
        rnn_num_layers=p_cfg.rnn_num_layers,
        head_hidden_dim=p_cfg.head_hidden_dim,
        img_obs=p_cfg.img_obs,
        dtype=jnp.bfloat16 if p_cfg.enable_bf16 else None,
    )
    params_p = load_p_params(args.p_checkpoint, net_p, env, env_params)

    # 3. Observer (O) Setup
    o_cfg = OModelCfg(
        fov_size=args.o_fov,
        obs_emb_dim=args.fp_emb,
        rnn_hidden_dim=args.fp_rnn,
        num_actions=env.num_actions(env_params)
    )
    
    # Use the factory from tom_nn to select the correct class
    # Note: we need a minimal config dict
    config = vars(args)
    rng = jax.random.key(args.seed)
    net_o, params_o = create_model(args.model_type, config, rng)
    
    # Load parameters
    if len(args.checkpoint) > 0:
        params_o = load_o_params(args.checkpoint, net_o, o_cfg, args.model_type)
    else:
        print("No TP checkpoint provided, using randomly initialized parameters.")

    # 4. Run
    run_dual_rollout(
        env, env_params, 
        net_p, params_p, 
        net_o, params_o,
        episodes=args.episodes,
        seed=args.seed,
        out_dir=args.vid_out_dir,
        fov_size=args.o_fov,
        model_type=args.model_type,
    )

if __name__ == "__main__":
    jax.config.update("jax_threefry_partitionable", True)
    main()