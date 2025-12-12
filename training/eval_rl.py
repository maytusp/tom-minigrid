"""
- Loads params from a .msgpack checkpoint
- Rebuilds the ActorCriticRNN
- Runs evaluation episodes
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

from .utils import rollout, rollout_with_obs, crop_fov_from_allocentric_rgb, _dir_to_id, crop_fov_symbolic_allocentric

from .nn import ActorCriticRNN

import xminigrid
from xminigrid.environment import EnvParams
from xminigrid.wrappers import AllocentricObservationWrapper, GymAutoResetWrapper
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
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1
    head_hidden_dim: int = 128
    img_obs: bool = False
    enable_bf16: bool = False
    use_color: bool = True # enable or disable color channel
    direction_obs = False 



def build_env(env_id: str, img_obs: bool, benchmark_id: str | None = None, ruleset_id: int | None = None):
    env, env_params = xminigrid.make(env_id)
    env = GymAutoResetWrapper(env)
    env = AllocentricObservationWrapper(env)

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
    max_videos: int = 20,
    max_steps: int = 1000,
    seed: int = 0,
    out_dir: str = "trajs",
    enable_bf16: bool = False,
    use_observer_frame: bool = False, # Observer frame is used if environment is bigger than observer FOV
    observer_r: int = None,
    observer_c: int = None,
    fov_size: int = 9,
    fov_dir: str = "up",
):
    os.makedirs(out_dir, exist_ok=True)
    curr_num_vids = 0 # count video
    ts = TrainState.create(apply_fn=net.apply, params=params, tx=optax.sgd(0.0))
    h0 = net.initialize_carry(batch_size=1)
    if enable_bf16:
        h0 = h0.astype(jnp.bfloat16)

    dir_id = _dir_to_id(fov_dir)

    @jax.jit
    def run_one(rng):
        out = rollout_with_obs(rng, env, env_params, ts, h0, max_steps=max_steps)
        # world RGB (allocentric)
        '''Render function get grid state (without agent) and agent state (position + direction), not agent observation which includes itself'''
        world_frames = jax.vmap(render_grid_allocentric)(out.grid_seq, out.agent_seq)  # [T,Hpx,Wpx,3] uint8

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
        
        if use_observer_frame:
            obs_rgb_frames = jax.vmap(_crop_rgb)(world_frames, out.allo_obs_seq)  # [T, fpx, fpx, 3]
            # observer SYMBOLIC crop (cell space), then simple colorization for MP4
            obs_sym_patches = jax.vmap(
                lambda grid_sym: crop_fov_symbolic_allocentric(
                    grid_sym=grid_sym, r=observer_r, c=observer_c, view_size=fov_size, dir_id=dir_id
                )
            )(out.allo_obs_seq)  # [T, V, V, C]

            return world_frames, obs_rgb_frames, obs_sym_patches, out.length - 1
        else:
            return world_frames, out.allo_obs_seq, out.length - 1

    rng = jax.random.key(seed)
    for ep in range(episodes):
        rng, sub = jax.random.split(rng)
        if use_observer_frame:
            world_dev, obs_rgb_dev, obs_sym_dev, T = run_one(sub)
        else:
            world_dev, obs_sym_dev, T = run_one(sub)

        T_int = int(np.asarray(T))

        world_np = np.asarray(world_dev[:T_int])

        obs_sym_np = np.asarray(obs_sym_dev[:T_int])        # raw symbolic [T,V,V,C]

        # paths
        out_world_mp4 = os.path.join(out_dir, f"ep_world_rgb_{ep:03d}.mp4")
        out_obs_rgb_mp4 = os.path.join(out_dir, f"ep_{ep:03d}_observer_rgb.mp4")
        out_obs_sym_npz = os.path.join(out_dir, f"ep_{ep:03d}_observer_sym.npz")
        if curr_num_vids < max_videos:
            # write videos
            if use_observer_frame:
                obs_rgb_np = np.asarray(obs_rgb_dev[:T_int])
                imageio.mimsave(out_obs_rgb_mp4, obs_rgb_np, fps=10)
                imageio.mimsave(out_world_mp4, world_np, fps=10) # save world frame seaprately
            else:
                imageio.mimsave(out_obs_rgb_mp4, world_np, fps=10)
            
            curr_num_vids += 1

        

        # write raw arrays
        np.savez_compressed(
            out_obs_sym_npz,
            sym=obs_sym_np,            # [T, V, V, C] exact symbolic crop per frame
            meta=dict(
                observer_r=observer_r,
                observer_c=observer_c,
                fov_size=fov_size,
                fov_dir=fov_dir,
                channels="[..., entity, color]",
            ),
        )

        # print(f"[record] {out_world_mp4}         frames={T_int} size={world_np.shape[1]}x{world_np.shape[2]}")
        # print(f"[record] {out_obs_rgb_mp4}      frames={T_int} size={obs_rgb_np.shape[1]}x{obs_rgb_np.shape[2]}")
        print(f"[record] {out_obs_sym_npz}      saved raw symbolic crops")



def eval_with_rollout(env, env_params, net, params, episodes: int, seed: int, enable_bf16: bool = False):
    '''
    Fast evaluation without saving anything.
    '''
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
    parser.add_argument("--checkpoint", type=str, default="checkpoints/MiniGrid-Protagonist-ProcGen-9x9vs9/MiniGrid-Protagonist-ProcGen-9x9vs9-ppo_final.msgpack")
    parser.add_argument("--env_id", type=str, default="MiniGrid-Protagonist-ProcGen-9x9vs9")
    parser.add_argument("--vid_out_dir", type=str, default="logs/trajs/MiniGrid-Protagonist-ProcGen-9x9vs9")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()

    # Build config & env
    cfg = ModelCfg()

    env, env_params = build_env(args.env_id, cfg.img_obs)
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
                        episodes=args.episodes, max_steps=1000, seed=0,
                        out_dir=args.vid_out_dir)
    # eval_with_rollout(env, env_params, net, params, episodes=args.episodes, seed=args.seed)

if __name__ == "__main__":
    # This flag matches your training script default and will become default in newer JAX versions.
    jax.config.update("jax_threefry_partitionable", True)
    main()
