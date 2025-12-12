# Record Third-Person Observation (Not used anymore because we will use observer with 9x9 and it matches the environment size)
# Input: Observer row,column, FOV size and FOV direction
# The observer sees the protagonist plays.
#!/usr/bin/env python3
"""
Minimal test loader & evaluator for the PPO-RNN model.

- Loads params from a .msgpack checkpoint saved via flax.serialization.to_bytes(...)
- Rebuilds the ActorCriticRNN with the SAME hyperparams used in training
- Runs a few evaluation episodes and prints average return/length

Usage:
    python test_load_eval.py \
      --checkpoint ./checkpoints/ppo_final.msgpack \
      --env_id MiniGrid-SwapEmpty-9x9 \
      --episodes 10 \
      --seed 1
"""

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np

import jax
import jax, jax.numpy as jnp, optax
from jax import lax, jit, device_put
from flax.training.train_state import TrainState
from flax.serialization import from_bytes, msgpack_restore, from_state_dict
from flax.core import freeze, unfreeze
from flax import traverse_util

from utils import rollout, rollout_with_obs, crop_fov_from_allocentric_rgb, _dir_to_id, crop_fov_symbolic_allocentric, symbol_to_color_rgb

from nn import ActorCriticRNN

import xminigrid
from xminigrid.environment import EnvParams
from xminigrid.wrappers import DirectionObservationWrapper, GymAutoResetWrapper
from xminigrid.experimental.img_obs import RGBImgObservationWrapper, _render_obs, render_grid_allocentric

from pathlib import Path
from typing import Optional
import imageio.v2 as imageio
from tqdm import trange
from imageio_ffmpeg import write_frames

# Match training default unless you changed them there.
@dataclass
class ModelCfg:
    obs_emb_dim: int = 16
    action_emb_dim: int = 16
    rnn_hidden_dim: int = 1024
    rnn_num_layers: int = 1
    head_hidden_dim: int = 256
    img_obs: bool = False
    enable_bf16: bool = False

def build_env(env_id: str, img_obs: bool, benchmark_id: str | None = None, ruleset_id: int | None = None):
    env, env_params = xminigrid.make(env_id)
    env = GymAutoResetWrapper(env)
    env = DirectionObservationWrapper(env)

    if benchmark_id is not None:
        assert "XLand-MiniGrid" in env_id, "Benchmarks should be used only with XLand environments."
        assert ruleset_id is not None, "Ruleset ID must be provided when using a benchmark."
        benchmark = xminigrid.load_benchmark(benchmark_id)
        env_params = env_params.replace(ruleset=benchmark.get_ruleset(ruleset_id))

    if img_obs:
        env = RGBImgObservationWrapper(env)

    return env, env_params

def load_params(checkpoint_path: str, net, env, env_params, cfg):
    # build target variables & params collection for shape checking
    shapes = env.observation_shape(env_params)
    init_obs = {
        "obs_img": jnp.zeros((1, 1, *shapes["img"])),
        "obs_dir": jnp.zeros((1, 1, shapes["direction"])),
        "prev_action": jnp.zeros((1, 1), dtype=jnp.int32),
        "prev_reward": jnp.zeros((1, 1)),
    }
    init_hstate = net.initialize_carry(batch_size=1)
    rng = jax.random.key(0)
    target_vars = net.init(rng, init_obs, init_hstate)   # {'params': ...}
    target_params = target_vars["params"]

    with open(checkpoint_path, "rb") as f:
        raw = msgpack_restore(f.read())

    raw_params = raw["params"] if (isinstance(raw, dict) and "params" in raw) else raw
    loaded_params = from_state_dict(target_params, raw_params)

    print(f"[loader] strict restore OK from {checkpoint_path}")
    return freeze({"params": loaded_params})  # <-- return variables dict


def record_vid_with_rollout_jax(
    env,
    env_params,
    net,
    params,
    *,
    episodes: int = 1,
    max_steps: int = 1000,
    seed: int = 0,
    out_dir: str = "trajs",
    enable_bf16: bool = False,
    # observer settings
    observer_r: int = 6,
    observer_c: int = 6,
    fov_size: int = 7,
    fov_dir: str = "up",
):
    os.makedirs(out_dir, exist_ok=True)

    ts = TrainState.create(apply_fn=net.apply, params=params, tx=optax.sgd(0.0))
    h0 = net.initialize_carry(batch_size=1)
    if enable_bf16:
        h0 = h0.astype(jnp.bfloat16)

    dir_id = _dir_to_id(fov_dir)

    @jax.jit
    def run_one(rng):
        out = rollout_with_obs(rng, env, env_params, ts, h0, max_steps=max_steps)
        # world RGB (allocentric)
        world_frames = jax.vmap(render_grid_allocentric)(out.allo_obs_seq, out.agent_seq)  # [T,Hpx,Wpx,3] uint8

        # observer RGB crop
        def _crop_rgb(frame_rgb, grid_symbolic):
            Hc = grid_symbolic.shape[0]
            Wc = grid_symbolic.shape[1]
            return crop_fov_from_allocentric_rgb(
                frame_rgb=frame_rgb,
                grid_cells_h=Hc,
                grid_cells_w=Wc,
                r=observer_r,
                c=observer_c,
                view_size=fov_size,
                dir_id=dir_id,
            )
        obs_rgb_frames = jax.vmap(_crop_rgb)(world_frames, out.allo_obs_seq)  # [T, fpx, fpx, 3]

        # observer SYMBOLIC crop (cell space), then simple colorization for MP4
        obs_sym_patches = jax.vmap(
            lambda grid_sym: crop_fov_symbolic_allocentric(
                grid_sym=grid_sym, r=observer_r, c=observer_c, view_size=fov_size, dir_id=dir_id
            )
        )(out.allo_obs_seq)  # [T, V, V, C]

        obs_sym_vis = jax.vmap(symbol_to_color_rgb)(obs_sym_patches)  # [T, V, V, 3] uint8

        return world_frames, obs_rgb_frames, obs_sym_patches, obs_sym_vis, out.length - 1

    rng = jax.random.key(seed)
    for ep in range(episodes):
        rng, sub = jax.random.split(rng)
        world_dev, obs_rgb_dev, obs_sym_dev, obs_sym_vis_dev, T = run_one(sub)
        T_int = int(np.asarray(T))

        world_np = np.asarray(world_dev[:T_int])
        obs_rgb_np = np.asarray(obs_rgb_dev[:T_int])
        obs_sym_np = np.asarray(obs_sym_dev[:T_int])        # raw symbolic [T,V,V,C]
        obs_sym_vis_np = np.asarray(obs_sym_vis_dev[:T_int])  # colorized for MP4

        # paths
        out_world_mp4 = os.path.join(out_dir, f"ep_{ep:03d}.mp4")
        out_obs_rgb_mp4 = os.path.join(out_dir, f"ep_{ep:03d}_observer_rgb.mp4")
        out_obs_sym_npz = os.path.join(out_dir, f"ep_{ep:03d}_observer_sym.npz")

        # write videos
        imageio.mimsave(out_world_mp4, world_np, fps=10)
        imageio.mimsave(out_obs_rgb_mp4, obs_rgb_np, fps=10)

        # write raw arrays
        np.savez_compressed(
            out_obs_sym_npz,
            sym=obs_sym_np,            # [T, V, V, C] exact symbolic crop per frame
            meta=dict(
                observer_r=observer_r,
                observer_c=observer_c,
                fov_size=fov_size,
                fov_dir=fov_dir,
                channels="[..., entity, color] (if your grid uses that convention)",
            ),
        )

        print(f"[record] {out_world_mp4}         frames={T_int} size={world_np.shape[1]}x{world_np.shape[2]}")
        print(f"[record] {out_obs_rgb_mp4}      frames={T_int} size={obs_rgb_np.shape[1]}x{obs_rgb_np.shape[2]}")
        print(f"[record] {out_obs_sym_npz}      saved raw symbolic crops")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/MiniGrid-ToM-TwoRoomsNoSwap-13x13-ppo_final.msgpack")
    parser.add_argument("--env_id", type=str, default="MiniGrid-ToM-SmallRoomsNoSwap-13x13") # TwoRoomsNoSwap-13x13"
    parser.add_argument("--vid_out_dir", type=str, default="videos/belief-test/MiniGrid-ToM-SmallRoomsNoSwap-13x13")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--img_obs", action="store_true", help="Use image observations (must match training)")
    parser.add_argument("--obs_emb_dim", type=int, default=16)
    parser.add_argument("--action_emb_dim", type=int, default=16)
    parser.add_argument("--rnn_hidden_dim", type=int, default=1024)
    parser.add_argument("--rnn_num_layers", type=int, default=1)
    parser.add_argument("--head_hidden_dim", type=int, default=256)
    parser.add_argument("--enable_bf16", action="store_true")
    parser.add_argument("--benchmark_id", type=str, default=None)
    parser.add_argument("--ruleset_id", type=int, default=None)
    parser.add_argument("--observer_r", type=int, default=9, help="Observer row (cell index) in allocentric grid")
    parser.add_argument("--observer_c", type=int, default=6, help="Observer col (cell index) in allocentric grid")
    parser.add_argument("--fov_size", type=int, default=7, help="Square FOV size in cells (odd recommended)")
    parser.add_argument("--fov_dir", type=str, default="up", choices=["up", "down", "bottom", "left", "right"],
                        help="Observer FOV direction in allocentric frame")
    args = parser.parse_args()

    # Build config & env
    cfg = ModelCfg(
        obs_emb_dim=args.obs_emb_dim,
        action_emb_dim=args.action_emb_dim,
        rnn_hidden_dim=args.rnn_hidden_dim,
        rnn_num_layers=args.rnn_num_layers,
        head_hidden_dim=args.head_hidden_dim,
        img_obs=args.img_obs,
        enable_bf16=args.enable_bf16,
    )

    env, env_params = build_env(args.env_id, cfg.img_obs, args.benchmark_id, args.ruleset_id)
    shapes = env.observation_shape(env_params)
    # print(f"shapes {shapes}")
    # Rebuild the exact network architecture
    net = ActorCriticRNN(
        num_actions=env.num_actions(env_params),
        obs_emb_dim=cfg.obs_emb_dim,
        action_emb_dim=cfg.action_emb_dim,
        rnn_hidden_dim=cfg.rnn_hidden_dim,
        rnn_num_layers=cfg.rnn_num_layers,
        head_hidden_dim=cfg.head_hidden_dim,
        img_obs=cfg.img_obs,
        dtype=jnp.bfloat16 if cfg.enable_bf16 else None,
    )

    # Load parameters from checkpoint
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    params = load_params(args.checkpoint, net, env, env_params, cfg)
    print(f"Loaded params from: {args.checkpoint}")

    record_vid_with_rollout_jax(
        env, env_params, net, params,
        episodes=10, max_steps=1000, seed=0,
        out_dir=args.vid_out_dir, enable_bf16=args.enable_bf16,
        observer_r=args.observer_r,
        observer_c=args.observer_c,
        fov_size=args.fov_size,
        fov_dir=args.fov_dir,
    )


if __name__ == "__main__":
    # This flag matches your training script default and will become default in newer JAX versions.
    jax.config.update("jax_threefry_partitionable", True)
    main()
