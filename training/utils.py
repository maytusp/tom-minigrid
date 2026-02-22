# utilities for PPO training and evaluation
import jax
from jax import lax
import jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState

from torch.utils.data import Dataset, DataLoader

from xminigrid.environment import Environment, EnvParams

from typing import NamedTuple

import numpy as np
import os
import glob

# Training stuff
class Transition(struct.PyTreeNode):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    # for obs
    obs: jax.Array
    dir: jax.Array
    # for rnn policy
    prev_action: jax.Array
    prev_reward: jax.Array


def calculate_gae(
    transitions: Transition,
    last_val: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    # single iteration for the loop
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        delta = transition.reward + gamma * next_value * (1 - transition.done) - transition.value
        gae = delta + gamma * gae_lambda * (1 - transition.done) * gae
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        transitions,
        reverse=True,
    )
    # advantages and values (Q)
    return advantages, advantages + transitions.value


def ppo_update_networks(
    train_state: TrainState,
    transitions: Transition,
    init_hstate: jax.Array,
    advantages: jax.Array,
    targets: jax.Array,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
):
    # NORMALIZE ADVANTAGES
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def _loss_fn(params):
        # RERUN NETWORK
        dist, value, _ = train_state.apply_fn(
            params,
            {
                # [batch_size, seq_len, ...]
                "obs_img": transitions.obs,
                "obs_dir": transitions.dir,
                "prev_action": transitions.prev_action,
                "prev_reward": transitions.prev_reward,
            },
            init_hstate,
        )
        log_prob = dist.log_prob(transitions.action)

        # CALCULATE VALUE LOSS
        value_pred_clipped = transitions.value + (value - transitions.value).clip(-clip_eps, clip_eps)
        value_loss = jnp.square(value - targets)
        value_loss_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_loss, value_loss_clipped).mean()

        # TODO: ablate this!
        # value_loss = jnp.square(value - targets).mean()

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - transitions.log_prob)
        actor_loss1 = advantages * ratio
        actor_loss2 = advantages * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()
        entropy = dist.entropy().mean()

        total_loss = actor_loss + vf_coef * value_loss - ent_coef * entropy
        return total_loss, (value_loss, actor_loss, entropy)

    (loss, (vloss, aloss, entropy)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params)
    (loss, vloss, aloss, entropy, grads) = jax.lax.pmean((loss, vloss, aloss, entropy, grads), axis_name="devices")
    train_state = train_state.apply_gradients(grads=grads)
    update_info = {
        "total_loss": loss,
        "value_loss": vloss,
        "actor_loss": aloss,
        "entropy": entropy,
    }
    return train_state, update_info


# for evaluation (evaluate for N consecutive episodes, sum rewards)
# N=1 single task, N>1 for meta-RL
class RolloutStats(struct.PyTreeNode):
    reward: jax.Array = jnp.asarray(0.0)
    length: jax.Array = jnp.asarray(0)
    episodes: jax.Array = jnp.asarray(0)


def rollout(
    rng: jax.Array,
    env: Environment,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: jax.Array,
    num_consecutive_episodes: int = 1,
) -> RolloutStats:
    def _cond_fn(carry):
        rng, stats, timestep, prev_action, prev_reward, hstate = carry
        return jnp.less(stats.episodes, num_consecutive_episodes)

    def _body_fn(carry):
        rng, stats, timestep, prev_action, prev_reward, hstate = carry

        rng, _rng = jax.random.split(rng)
        dist, _, hstate = train_state.apply_fn(
            train_state.params,
            {
                "obs_img": timestep.observation["p_img"][None, None, ...],
                "obs_dir": timestep.observation["direction"][None, None, ...],
                "prev_action": prev_action[None, None, ...],
                "prev_reward": prev_reward[None, None, ...],
            },
            hstate,
        )
        action = dist.sample(seed=_rng).squeeze()
        timestep = env.step(env_params, timestep, action)

        stats = stats.replace(
            reward=stats.reward + timestep.reward,
            length=stats.length + 1,
            episodes=stats.episodes + timestep.last(),
        )
        carry = (rng, stats, timestep, action, timestep.reward, hstate)
        return carry

    timestep = env.reset(env_params, rng)
    prev_action = jnp.asarray(0)
    prev_reward = jnp.asarray(0)
    init_carry = (rng, RolloutStats(), timestep, prev_action, prev_reward, init_hstate)

    final_carry = jax.lax.while_loop(_cond_fn, _body_fn, init_val=init_carry)
    return final_carry[1]


class RolloutWithObs(NamedTuple):
    stats: RolloutStats
    obs_seq: jax.Array    # [T_max, *obs_img_shape], symbolic grid per step
    o_obs_seq: jax.Array # [T_max, *o_obs_img_shape], symbolic grid per step
    agent_seq: jax.Array # [T_max, 2+1] agent's position + direction
    grid_seq: jax.Array  # Grid state, similar to o_obs but exclude agent from it
    action_seq: jax.Array # [T_max] agent's actions (int32)
    length: jax.Array     # scalar int32, actual T (<= T_max)

def rollout_with_obs(
    rng: jax.Array,
    env: Environment,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: jax.Array,
    *,
    max_steps: int = 1000,
) -> RolloutWithObs:
    """
    WARNING: Allocentric observation from state.grid does not contain agent. Only state.observation has agent.
    Runs a SINGLE episode fully in JAX (lax.while_loop), collecting the symbolic
    obs["p_img"] at each step into a preallocated buffer of length max_steps.

    Returns:
      - stats: same fields as your RolloutStats (reward, length, episodes)
      - obs_seq: [max_steps, *obs_img_shape] symbolic frames (not RGB)
      - length: actual number of steps taken (<= max_steps)
    """
    # Reset once (single episode)
    timestep0 = env.reset(env_params, rng)

    # Shapes/dtypes for the obs buffer
    obs_img0 = timestep0.observation["p_img"]
    allo_img0 = timestep0.observation["o_img"]
    grid0 = timestep0.state.grid
    agent_pos0 = timestep0.state.agent.position
    agent_dir0 = timestep0.state.agent.direction
    
    obs_buf0 = jnp.zeros((max_steps, *obs_img0.shape), obs_img0.dtype)
    o_obs_buf0 = jnp.zeros((max_steps, *allo_img0.shape), allo_img0.dtype)
    agent_buf0 = jnp.zeros((max_steps, agent_pos0.shape[0]+1), agent_pos0.dtype)
    grid_buf0 = jnp.zeros((max_steps, *grid0.shape), grid0.dtype)
    action_buf0 = jnp.zeros((max_steps,), dtype=jnp.int32)

    # Initial prev_* and bookkeeping
    prev_action0 = jnp.asarray(0)
    prev_reward0 = jnp.asarray(0)
    stats0 = RolloutStats()  # same as your rollout
    h0 = init_hstate
    step0 = jnp.int32(0)

    # Write the very first frame (pre-step) at index 0
    obs_buf0 = obs_buf0.at[0].set(obs_img0)
    o_obs_buf0 = o_obs_buf0.at[0].set(allo_img0)
    agent_buf0 = agent_buf0.at[0].set(jnp.concatenate([agent_pos0, agent_dir0[None]], axis=0))
    grid_buf0 = grid_buf0.at[0].set(grid0)

    def cond_fn(carry):
        rng, stats, timestep, prev_action, prev_reward, h, obs_buf, o_obs_buf, agent_buf, grid_buf, action_buf, step = carry
        # Continue while not done with episode (episodes<1) AND under max_steps-1
        # We allow writing index 0..length, so we stop when the *next* write would overflow.
        return jnp.logical_and(stats.episodes < 1, step < max_steps - 1)

    def body_fn(carry):
        rng, stats, timestep, prev_action, prev_reward, h, obs_buf, o_obs_buf, agent_buf, grid_buf, action_buf, step = carry

        rng, _rng = jax.random.split(rng)
        # Forward policy on symbolic obs
        dist, _, h = train_state.apply_fn(
            train_state.params,
            {
                "obs_img": timestep.observation["p_img"][None, None, ...],
                "obs_dir": timestep.observation["direction"][None, None, ...],
                "prev_action": prev_action[None, None, ...],
                "prev_reward": prev_reward[None, None, ...],
            },
            h,
        )

        action = dist.sample(seed=_rng).squeeze()
        timestep_next = env.step(env_params, timestep, action)

        # Update stats like your rollout
        stats_next = stats.replace(
            reward=stats.reward + timestep_next.reward,
            length=stats.length + 1,
            episodes=stats.episodes + timestep_next.last(),
        )

        # Write the next pre-step frame (the observation we’ll act on next)
        # i.e., store obs for timestep_next at index (step+1)
        action_buf = action_buf.at[step].set(action)
        obs_buf = obs_buf.at[step + 1].set(timestep_next.observation["p_img"])
        o_obs_buf = o_obs_buf.at[step + 1].set(timestep_next.observation["o_img"])
        pos = timestep_next.state.agent.position        # shape (2,)
        direction = jnp.asarray(timestep_next.state.agent.direction)  # shape ()
        agent_buf = agent_buf.at[step + 1].set(
            jnp.concatenate([pos, direction[None]], axis=0)           # shape (3,)
        )
        grid_buf = grid_buf.at[step+1].set(timestep_next.state.grid)
        return (
            rng,
            stats_next,
            timestep_next,
            action,
            timestep_next.reward,
            h,
            obs_buf,
            o_obs_buf,
            agent_buf,
            grid_buf,
            action_buf,
            step + 1,
        )

    final = lax.while_loop(
            cond_fn,
            body_fn,
            (rng, stats0, timestep0, prev_action0, prev_reward0, h0, obs_buf0, o_obs_buf0, agent_buf0, grid_buf0, action_buf0, step0),
        )

    _, stats_f, _, _, _, _, obs_buf_f, o_obs_buf_f, agent_buf_f, grid_buf_f, action_buf_f, step_f = final

    # The actual usable frames run from [0 .. stats_f.length] inclusive of the first frame,
    # but since we wrote one frame per step (including the last post-step obs at index=length),
    # the effective count to keep is `stats_f.length + 1`, capped by max_steps.
    # For downstream convenience, return `length = stats_f.length + 1`.
    length_plus_one = jnp.minimum(step_f + 1, jnp.int32(max_steps))

    return RolloutWithObs(
        stats=stats_f,
        obs_seq=obs_buf_f,
        o_obs_seq=o_obs_buf_f,
        agent_seq=agent_buf_f,
        grid_seq=grid_buf_f,
        action_seq=action_buf_f,
        length=length_plus_one,
    )


def _dir_to_id(d: str) -> int:
    # We normalize "bottom" to "down"
    if d == "bottom":
        d = "down"
    mapping = {"up": 0, "right": 1, "down": 2, "left": 3}
    if d not in mapping:
        raise ValueError(f"Unknown fov_dir: {d}")
    return mapping[d]


def crop_fov_from_allocentric_rgb(
    frame_rgb: jnp.ndarray,     # [Hpx, Wpx, 3] uint8
    grid_cells_h: int,          # rows in grid (cells) — pass Python ints
    grid_cells_w: int,          # cols in grid (cells)
    r: int, c: int,             # observer row/col (cells)
    view_size: int,             # FOV (cells, square)
    dir_id: int,                # 0=up, 1=right, 2=down, 3=left
) -> jnp.ndarray:
    Hpx, Wpx = frame_rgb.shape[0], frame_rgb.shape[1]
    # pixels-per-cell from static shapes -> Python ints
    tile_h = Hpx // int(grid_cells_h)
    tile_w = Wpx // int(grid_cells_w)
    tile   = min(tile_h, tile_w)

    half = view_size // 2
    if dir_id == 0:      # up
        r0, r1 = r - (view_size - 1), r + 1
        c0, c1 = c - half, c + half + 1
        rot_k = 0
    elif dir_id == 2:    # down
        r0, r1 = r, r + view_size
        c0, c1 = c - half, c + half + 1
        rot_k = 2
    elif dir_id == 3:    # left
        r0, r1 = r - half, r + half + 1
        c0, c1 = c - (view_size - 1), c + 1
        rot_k = -1
    else:                # right
        r0, r1 = r - half, r + half + 1
        c0, c1 = c, c + view_size
        rot_k = 1

    # Desired output size in pixels (static under jit)
    out_h = (r1 - r0) * tile
    out_w = (c1 - c0) * tile

    # Clamp slice size so it always fits the source image (must be static ints)
    slice_h = min(out_h, Hpx)
    slice_w = min(out_w, Wpx)

    # Compute start indices (dynamic) such that start + size <= dim
    pr0 = r0 * tile
    pc0 = c0 * tile
    start_r = jnp.clip(pr0, 0, Hpx - slice_h)
    start_c = jnp.clip(pc0, 0, Wpx - slice_w)

    # Extract valid portion with dynamic_slice (sizes must be static)
    patch_valid = lax.dynamic_slice(frame_rgb, (start_r, start_c, 0), (slice_h, slice_w, 3))

    # Prepare zero canvas of desired FOV size and paste the valid slice at the right offset
    out = jnp.zeros((out_h, out_w, 3), dtype=frame_rgb.dtype)
    dr = start_r - pr0   # amount we had to shift down due to clamping (>=0)
    dc = start_c - pc0   # amount we had to shift right due to clamping (>=0)
    out = lax.dynamic_update_slice(out, patch_valid, (dr, dc, 0))

    # Rotate so "forward" is up
    if rot_k != 0:
        out = jnp.rot90(out, k=rot_k, axes=(0, 1))
    return out

def crop_fov_symbolic_allocentric(
    grid_sym: jnp.ndarray,   # [Hc, Wc, C]
    r: int, c: int,
    view_size: int,
    dir_id: int,             # 0=up, 1=right, 2=down, 3=left
) -> jnp.ndarray:
    Hc, Wc, C = grid_sym.shape
    half = view_size // 2

    if dir_id == 0:      # up
        r0, r1 = r - (view_size - 1), r + 1
        c0, c1 = c - half, c + half + 1
        rot_k = 0
    elif dir_id == 2:    # down
        r0, r1 = r, r + view_size
        c0, c1 = c - half, c + half + 1
        rot_k = 2
    elif dir_id == 3:    # left
        r0, r1 = r - half, r + half + 1
        c0, c1 = c - (view_size - 1), c + 1
        rot_k = -1
    else:                # right
        r0, r1 = r - half, r + half + 1
        c0, c1 = c, c + view_size
        rot_k = 1

    out_h = (r1 - r0)
    out_w = (c1 - c0)

    slice_h = min(out_h, Hc)
    slice_w = min(out_w, Wc)

    start_r = jnp.clip(r0, 0, Hc - slice_h)
    start_c = jnp.clip(c0, 0, Wc - slice_w)

    patch_valid = lax.dynamic_slice(grid_sym, (start_r, start_c, 0), (slice_h, slice_w, C))

    out = jnp.zeros((out_h, out_w, C), dtype=grid_sym.dtype)
    dr = start_r - r0
    dc = start_c - c0
    out = lax.dynamic_update_slice(out, patch_valid, (dr, dc, 0))

    if rot_k != 0:
        out = jnp.rot90(out, k=rot_k, axes=(0, 1))
    return out



def symbol_to_color_rgb(
    sym_patch: jnp.ndarray,  # [V,V,C] (assume channel 0 is "entity" id)
) -> jnp.ndarray:
    """
    Lightweight visualization: map entity IDs (channel 0) to grayscale RGB.
    This is only for quick viewing; .npz will store the exact symbolic values.
    """
    ent = sym_patch[..., 0].astype(jnp.int32)
    max_id = jnp.maximum(ent.max(), 1)
    gray = (ent * (255 // max_id)).astype(jnp.uint8)
    rgb = jnp.stack([gray, gray, gray], axis=-1)
    return rgb



def append_episode_flat(buffers, obs_sym_np):
    # obs_sym_np: [T, V, V, C] (your observer symbolic crop)
    # Build (x_t, x_{t+1}) pairs:
    X = obs_sym_np[:-1]               # [T-1, V, V, C]
    Y = obs_sym_np[1:]                # [T-1, V, V, C]
    N = X.shape[0]
    ep_id = np.int32(buffers["next_episode_id"])
    ep = np.full((N,), ep_id, dtype=np.int32)
    t  = np.arange(N, dtype=np.int32)

    # Append to buffers (lists)
    buffers["X"].append(X)
    buffers["Y"].append(Y)
    buffers["ep"].append(ep)
    buffers["t"].append(t)
    buffers["next_episode_id"] += 1
    return buffers

def write_shard_npz(buffers, out_dir, shard_idx, meta):
    X = np.concatenate(buffers["X"], axis=0) if buffers["X"] else np.zeros((0,), dtype=np.uint8)
    Y = np.concatenate(buffers["Y"], axis=0) if buffers["Y"] else np.zeros((0,), dtype=np.uint8)
    ep = np.concatenate(buffers["ep"], axis=0) if buffers["ep"] else np.zeros((0,), dtype=np.int32)
    t  = np.concatenate(buffers["t"], axis=0) if buffers["t"] else np.zeros((0,), dtype=np.int32)

    path = os.path.join(out_dir, f"observer_sym_shard_{shard_idx:04d}.npz")
    np.savez_compressed(path, X=X, Y=Y, ep=ep, t=t, meta=meta)
    return path



DIR_TO_IDX = {"right": 0, "down": 1, "left": 2, "up": 3}

def get_direction_one_hot(dir_str: str) -> np.ndarray:
    """Returns a [4] float array for the direction."""
    idx = DIR_TO_IDX.get(dir_str.lower(), 0)
    one_hot = np.zeros(4, dtype=np.float32)
    one_hot[idx] = 1.0
    return one_hot

class NpzEpisodeDataset(Dataset):
    def __init__(self, data_dir: str, max_files: int = None):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if max_files:
            self.files = self.files[:max_files]
        print(f"[Dataset] Found {len(self.files)} episodes in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            with np.load(path, allow_pickle=True) as data:
                # 1. Observation: [T, V, V, C]
                obs_seq = data['o_sym'].astype(np.int32)
                T = obs_seq.shape[0]

                # 2. Meta / Direction
                meta = data['meta'].item() if data['meta'].shape == () else data['meta']
                fov_dir = meta.get('fov_dir', 'up')
                dir_vec = get_direction_one_hot(fov_dir)
                dir_seq = np.tile(dir_vec, (T, 1))

                # 3. Actions
                if 'action' in data:
                    action_seq = data['action'].astype(np.int32)
                else:
                    action_seq = np.zeros((T,), dtype=np.int32)

                # 4. Rewards
                if 'reward' in data:
                    reward_seq = data['reward'].astype(np.float32)
                else:
                    reward_seq = np.zeros((T,), dtype=np.float32)

                # 5. Targets (Next Action)
                if 'other_action' in data:
                    next_act = data['other_action'].astype(np.int32)
                else:
                    next_act = np.zeros((T,), dtype=np.int32) - 1
                
                # 6. Done
                done_seq = np.zeros((T,), dtype=np.float32)
                done_seq[-1] = 1.0 

            return {
                "o_obs": obs_seq,
                "dir": dir_seq,
                "act": action_seq,
                "rew": reward_seq,
                "next_act": next_act,
                "done": done_seq,
                "length": T,
                "file_id": os.path.basename(path)
            }
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return self.__getitem__((idx + 1) % len(self.files))

def pad_collate(batch):
    """Pads batch to the maximum length and aggregates metadata."""
    max_len = max(x['length'] for x in batch)
    B = len(batch)
    
    v_obs = batch[0]['o_obs']
    _, H, W, C = v_obs.shape
    
    out_obs = np.zeros((B, max_len, H, W, C), dtype=np.int32)
    out_dir = np.zeros((B, max_len, 4), dtype=np.float32)
    out_act = np.zeros((B, max_len), dtype=np.int32)
    out_rew = np.zeros((B, max_len), dtype=np.float32)
    out_next_act = np.zeros((B, max_len), dtype=np.int32)
    out_done = np.zeros((B, max_len), dtype=np.float32)
    mask_pad = np.zeros((B, max_len), dtype=np.float32)
    
    # New lists for metadata
    file_ids = [] 
    lengths = []

    for i, x in enumerate(batch):
        L = x['length']
        out_obs[i, :L] = x['o_obs']
        out_dir[i, :L] = x['dir']
        out_act[i, :L] = x['act']
        out_rew[i, :L] = x['rew']
        out_next_act[i, :L] = x['next_act']
        out_done[i, :L] = x['done']
        mask_pad[i, :L] = 1.0
        
        # Collect metadata
        file_ids.append(x['file_id']) 
        lengths.append(L)             
        
    return {
        "o_obs": out_obs,
        "dir": out_dir,
        "act": out_act,
        "rew": out_rew,
        "next_act": out_next_act,
        "done": out_done,
        "mask_pad": mask_pad,
        "file_ids": file_ids,         
        "length": np.array(lengths),
    }
