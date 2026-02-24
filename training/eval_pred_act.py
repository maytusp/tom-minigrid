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
from .utils import _dir_to_id, crop_fov_from_allocentric_rgb, crop_fov_symbolic_allocentric

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
    max_steps: int = 1000,
    seed: int = 0,
    out_dir: str = "trajs_pred",
    # observer_r: int = 8,
    # observer_c: int = 5,
    # fov_dir: str = "up",
    observer_r: int = 5,
    observer_c: int = 1,
    fov_dir = "right",
    fov_size: int = 9,
    
):
    os.makedirs(out_dir, exist_ok=True)
    
    # P Init
    h0_p = net_p.initialize_carry(batch_size=1)
    
    # O Init (Dual model returns 2 states: h_fp, h_tp)
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
            obs_o = crop_fov_symbolic_allocentric(
                grid_sym=obs_o, 
                r=observer_r, 
                c=observer_c, 
                view_size=fov_size, 
                dir_id=dir_id
            )
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

            logits_o, _, new_h_o_fp, new_h_o_tp = net_o.apply(
                {"params": params_o}, 
                in_o_fp, h_o_fp, in_o_tp, h_o_tp
            )

            probs_o = jax.nn.softmax(logits_o)
            entropy = -jnp.sum(probs_o * jnp.log(probs_o + 1e-6))
            action_o = jnp.argmax(logits_o, axis=-1).squeeze()

            # Check if we are in the "Predicted" phase (Star is gone)
            star_visible = jnp.any(obs_o[..., 0] == 12)
            
            # Print only when simulating (star gone) to avoid spam
            # The format string uses {x} where x is the value passed in ordered args
            jax.debug.print(
                "Step: Star={x} | Action={y} | Entropy={z:.3f} | Probs={p}",
                x=star_visible,
                y=action_o,
                z=entropy,
                p=probs_o[0, 0] # Print the probability distribution
            )

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
            
            outs = {
                "act_o_img": ts_act.observation["o_img"],
                "pred_o_img": ts_pred.observation["o_img"],
                "action_p": action_p,
                "action_o": action_o,
                "used_action": action_for_pred,
                "star_visible": star_visible,
                "reward_act": ts_act.reward,
                "done_act": ts_act.last(),
            }
            return new_carry, outs

        final_carry, scan_out = jax.lax.scan(step_fn, init_carry, None, length=max_steps)
        return scan_out

    def render_traj(scan_out, T):
        o_img_act = scan_out["act_o_img"][:T]
        o_img_pred = scan_out["pred_o_img"][:T]
        
        rgb_act = jax.vmap(_render)(o_img_act)
        rgb_pred = jax.vmap(_render)(o_img_pred)

        def _crop(rgb, sym):
            Hc, Wc = sym.shape[0], sym.shape[1]
            return crop_fov_from_allocentric_rgb(rgb, Hc, Wc, observer_r, observer_c, fov_size, dir_id)
        
        rgb_act_crop = jax.vmap(_crop)(rgb_act, o_img_act)
        rgb_pred_crop = jax.vmap(_crop)(rgb_pred, o_img_pred)
        
        B, H, W, C = rgb_act_crop.shape
        sep = jnp.ones((B, H, 2, C), dtype=rgb_act_crop.dtype) * 255
        
        combined = jnp.concatenate([rgb_act_crop, sep, rgb_pred_crop], axis=2)
        return combined

    rng = jax.random.key(seed)
    
    for ep in range(episodes):
        rng, sub_rng = jax.random.split(rng)
        out = run_one_episode(sub_rng)
        
        dones = np.array(out["done_act"])
        T = (np.argmax(dones) + 1) if dones.any() else max_steps
        
        reward = np.sum(out["reward_act"][:T])
        print(f"Episode {ep}: T={T}, Reward={reward}")
        
        video_frames = render_traj(out, T)
        video_np = np.array(video_frames)
        
        vid_path = os.path.join(out_dir, f"ep_{ep:03d}_pred_comparison.mp4")
        imageio.mimsave(vid_path, video_np, fps=5)
        
        np.savez(
            os.path.join(out_dir, f"ep_{ep:03d}_data.npz"),
            action_p=out["action_p"][:T],
            action_o=out["action_o"][:T],
            used_action=out["used_action"][:T],
            star_visible=out["star_visible"][:T]
        )
        print(f"Saved {vid_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p_checkpoint", type=str, default="./checkpoints/MiniGrid-ToM-TwoRoomsNoSwap-9x9vs9/MiniGrid-ToM-TwoRoomsNoSwap-9x9vs9-ppo_final.msgpack")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/observers/tworoom-noswap/tp-sr/checkpoint_49.msgpack", help="Observer checkpoint path")
    parser.add_argument("--model_type", type=str, default="third_person", choices=["third_person", "dual_perspective"])
    parser.add_argument("--use_sr", action="store_true", default=True, 
                            help="Enable Successor Representation prediction and loss.")
    parser.add_argument("--num_states", type=int, default=81, 
                        help="Total number of discrete states for the SR (update based on your grid size).")
    parser.add_argument("--env_id", type=str, default="MiniGrid-ToM-TwoRoomsSwap-9x9vs9")
    parser.add_argument("--vid_out_dir", type=str, default="logs/eval_pred_action/tp-sr/")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    
    # O args (must match training)
    parser.add_argument("--o_fov", type=int, default=9)

    parser.add_argument("--num_actions", type=int, default=6) 
    parser.add_argument("--fp_emb", type=int, default=16)
    parser.add_argument("--fp_rnn", type=int, default=256)
    parser.add_argument("--tp_emb", type=int, default=16)
    parser.add_argument("--tp_rnn", type=int, default=256)
    args = parser.parse_args()

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
    net_o, _ = create_model(args.model_type, config, rng)
    
    # Load parameters
    params_o = load_o_params(args.checkpoint, net_o, o_cfg, args.model_type)

    # 4. Run
    run_dual_rollout(
        env, env_params, 
        net_p, params_p, 
        net_o, params_o,
        episodes=args.episodes,
        seed=args.seed,
        out_dir=args.vid_out_dir,
        fov_size=args.o_fov
    )

if __name__ == "__main__":
    jax.config.update("jax_threefry_partitionable", True)
    main()