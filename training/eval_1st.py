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

import jax
import jax, jax.numpy as jnp, optax
from jax import jit, device_put

from flax.serialization import from_bytes
from flax.training.train_state import TrainState

from utils import rollout  # the same function used in training

from nn import ActorCriticRNN

import xminigrid
from xminigrid.environment import EnvParams
from xminigrid.wrappers import DirectionObservationWrapper, GymAutoResetWrapper

from pathlib import Path
from typing import Optional
import imageio.v2 as imageio
from tqdm import trange

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
        from xminigrid.experimental.img_obs import RGBImgObservationWrapper
        env = RGBImgObservationWrapper(env)

    return env, env_params

def load_params(checkpoint_path: str, net: ActorCriticRNN, env, env_params: EnvParams, cfg: ModelCfg):
    """
    Recreate the param tree with a dummy init, then fill it with checkpoint bytes.
    """
    # Create dummy inputs exactly like training (seq_len=1 per forward)
    shapes = env.observation_shape(env_params)
    init_obs = {
        "obs_img": jnp.zeros((1, 1, *shapes["img"])),
        "obs_dir": jnp.zeros((1, 1, shapes["direction"])),
        "prev_action": jnp.zeros((1, 1), dtype=jnp.int32),
        "prev_reward": jnp.zeros((1, 1)),
    }
    init_hstate = net.initialize_carry(batch_size=1)
    rng = jax.random.key(0)
    params_shape = net.init(rng, init_obs, init_hstate)  # param structure

    with open(checkpoint_path, "rb") as f:
        ckpt_bytes = f.read()

    params = from_bytes(params_shape, ckpt_bytes)  # fills the structure with saved weights
    return params


def save_eval_videos(
    env,
    env_params,
    net,
    params,
    *,
    num_videos: int = 10,
    max_steps: int = 1000,
    fps: int = 32,
    out_dir: str = "videos",
    seed: int = 0,
    enable_bf16: bool = False,
):
    """
    Runs `num_videos` single-episode rollouts and saves each episode as an MP4.
    Files: videos/ep_000.mp4, videos/ep_001.mp4, ..., videos/ep_009.mp4
    """

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    params = device_put(params)
    dtype = jnp.bfloat16 if enable_bf16 else None

    @jit
    def policy_step(params, carry, obs_img, obs_dir, prev_action, prev_reward, rng):
        # inputs need shape (B=1, 1, ...) to match training apply
        inputs = {
            "obs_img": obs_img[None, None, ...],       # (1,1,...)
            "obs_dir": obs_dir[None, None, ...],       # (1,1)
            "prev_action": prev_action[None, ...],     # (1,1)
            "prev_reward": prev_reward[None, ...],     # (1,1)
        }
        dist, value, carry = net.apply(params, inputs, carry)
        rng, sub = jax.random.split(rng)
        action = dist.sample(seed=sub).squeeze(1)      # (1,)
        return action, carry, rng
    base_key = jax.random.PRNGKey(seed)

    for ep in trange(num_videos, desc="Saving videos"):
        # Reset env
        rng = jax.random.fold_in(base_key, ep)
        timestep = env.reset(env_params, rng)

        carry = net.initialize_carry(batch_size=1)
        if dtype is not None:
            carry = carry.astype(dtype)

        prev_action = jnp.zeros((1,), dtype=jnp.int32)
        prev_reward = jnp.zeros((1,), dtype=jnp.float32)

        # Open writer
        video_path = os.path.join(out_dir, f"ep_{ep:03d}.mp4")
        images = []

        # First frame (at reset)
        frame0 = env.render(env_params, timestep)


        # Rollout
        for _ in range(max_steps):
            # Policy step on GPU
            action, carry, rng = policy_step(
                params,
                carry,
                timestep.observation["img"],
                timestep.observation["direction"],
                prev_action,               # shape (1,)
                prev_reward,               # shape (1,)
                rng,
            )

            # Step env (scalar action)
            timestep = env.step(env_params, timestep, int(action[0]))

            # Append frame
            frame = env.render(env_params, timestep)
            images.append(frame)

            # Prep next inputs (keep shape (1,))
            prev_action = action
            prev_reward = jnp.asarray([timestep.reward], dtype=jnp.float32)

            if bool(timestep.last()):
                break
        imageio.mimsave(video_path, images, fps=32, format="mp4")
        print(f"Saved {video_path}")

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
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ppo_step_15.msgpack")
    parser.add_argument("--env_id", type=str, default="MiniGrid-SwapEmpty-9x9")
    parser.add_argument("--episodes", type=int, default=100000)
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
    save_eval_videos(
        env,
        env_params,
        net,
        params,
        num_videos=10,
        max_steps=1000,
        fps=32,
        out_dir="videos",
        seed=args.seed,
        enable_bf16=args.enable_bf16,
    )
    # Run quick evaluation
    eval_with_rollout(env, env_params, net, params, episodes=args.episodes, seed=args.seed, enable_bf16=args.enable_bf16)

if __name__ == "__main__":
    # This flag matches your training script default and will become default in newer JAX versions.
    jax.config.update("jax_threefry_partitionable", True)
    main()
