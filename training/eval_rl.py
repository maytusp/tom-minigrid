"""
- Loads params from a .msgpack checkpoint
- Rebuilds the ActorCriticRNN
- Runs evaluation episodes

Usage:
python -m training.eval_rl \
      --checkpoint ./checkpoints/ppo_final.msgpack \
      --env_id MiniGrid-Protagonist-ProcGen-9x9vs9 \
      --episodes 1000 \
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
from flax.serialization import from_bytes, to_bytes, msgpack_restore, from_state_dict
from flax.core import freeze, unfreeze
from flax import traverse_util

from .utils import rollout, rollout_with_obs  # the same function used in training
from .nn import ActorCriticRNN

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

def record_with_rollout_jax(
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
):
    os.makedirs(out_dir, exist_ok=True)

    ts = TrainState.create(apply_fn=net.apply, params=params, tx=optax.sgd(0.0))
    h0 = net.initialize_carry(batch_size=1)
    if enable_bf16:
        h0 = h0.astype(jnp.bfloat16)

    @jax.jit
    def run_one(rng):
        out = rollout_with_obs(rng, env, env_params, ts, h0, max_steps=max_steps)
        # out.obs_seq: [T_max, *, *, *] symbolic grid
        # Render all frames on device
        frames = jax.vmap(render_grid_allocentric)(out.allo_obs_seq, out.agent_seq)  # [T_max, H, W, 3] uint8
        return frames, out.length  # length = (steps + 1) frames to keep

    rng = jax.random.key(seed)
    for ep in range(episodes):
        rng, sub = jax.random.split(rng)
        frames_dev, T = run_one(sub)
        T_int = int(np.asarray(T-1))
        frames = np.asarray(frames_dev[:T_int])  # single host copy
        out_path = os.path.join(out_dir, f"ep_{ep:03d}.mp4")
        imageio.mimsave(out_path, frames, fps=10)
        print(f"[record] {out_path}  frames={T_int}  size={frames.shape[1]}x{frames.shape[2]}")

def eval_with_rollout(env, env_params, net, params, episodes: int, seed: int, enable_bf16: bool = False):
    dtype = jnp.bfloat16 if enable_bf16 else None

    # Make a dummy TrainState so rollout can call `apply_fn`
    ts = TrainState.create(apply_fn=net.apply, params=params, tx=optax.sgd(0.0))

    # Hidden state for batch=1 (rollout handles time)
    init_hstate = net.initialize_carry(batch_size=1)
    if dtype is not None:
        init_hstate = init_hstate.astype(dtype)

    @jax.jit
    def eval_one(rng):
        stats = rollout(rng, env, env_params, ts, init_hstate, 1)
        # These are scalars in your setup; axis-less reduce is safe for 0-D or 1-D.
        ret = jnp.sum(stats.reward)
        length = jnp.sum(stats.length)
        return ret, length

    rng = jax.random.key(seed)
    rngs = jax.random.split(rng, episodes)

    # Vectorize + jit across episodes
    returns, lengths = jax.jit(jax.vmap(eval_one))(rngs)
    returns = jnp.asarray(returns); lengths = jnp.asarray(lengths)

    print("\n=== Eval Summary (JAX rollout) ===")
    print(f"Episodes: {episodes}")
    print(f"Avg return: {float(returns.mean()):.3f}")
    print(f"Avg length: {float(lengths.mean()):.3f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/MiniGrid-ToM-FourRoomsNoSwap-13x13-ppo_final.msgpack")
    parser.add_argument("--env_id", type=str, default="MiniGrid-ToM-FourRoomsNoSwap-13x13")
    parser.add_argument("--vid_out_dir", type=str, default="videos/MiniGrid-ToM-FourRoomsNoSwap-13x13")
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


    record_with_rollout_jax(env, env_params, net, params,
                        episodes=10, max_steps=1000, seed=0,
                        out_dir=args.vid_out_dir, enable_bf16=args.enable_bf16)
    eval_with_rollout(env, env_params, net, params, episodes=args.episodes, seed=args.seed, enable_bf16=args.enable_bf16)

if __name__ == "__main__":
    # This flag matches your training script default and will become default in newer JAX versions.
    jax.config.update("jax_threefry_partitionable", True)
    main()
