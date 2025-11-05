# utilities for PPO training and evaluation
import jax
from jax import lax
import jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState

from xminigrid.environment import Environment, EnvParams

from typing import NamedTuple


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
                "obs_img": timestep.observation["img"][None, None, ...],
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
    allo_obs_seq: jax.Array # [T_max, *allo_obs_img_shape], symbolic grid per step
    agent_seq: jax.Array # [T_max, 2+1] agent's position + direction
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
    Runs a SINGLE episode fully in JAX (lax.while_loop), collecting the symbolic
    obs["img"] at each step into a preallocated buffer of length max_steps.

    Returns:
      - stats: same fields as your RolloutStats (reward, length, episodes)
      - obs_seq: [max_steps, *obs_img_shape] symbolic frames (not RGB)
      - length: actual number of steps taken (<= max_steps)
    """
    # Reset once (single episode)
    timestep0 = env.reset(env_params, rng)

    # Shapes/dtypes for the obs buffer
    obs_img0 = timestep0.observation["img"]
    grid0 = timestep0.state.grid
    agent_pos0 = timestep0.state.agent.position
    agent_dir0 = timestep0.state.agent.direction
    
    obs_buf0 = jnp.zeros((max_steps, *obs_img0.shape), obs_img0.dtype)
    allo_obs_buf0 = jnp.zeros((max_steps, *grid0.shape), grid0.dtype)
    agent_buf0 = jnp.zeros((max_steps, agent_pos0.shape[0]+1), agent_pos0.dtype)

    # Initial prev_* and bookkeeping
    prev_action0 = jnp.asarray(0)
    prev_reward0 = jnp.asarray(0)
    stats0 = RolloutStats()  # same as your rollout
    h0 = init_hstate
    step0 = jnp.int32(0)

    # Write the very first frame (pre-step) at index 0
    obs_buf0 = obs_buf0.at[0].set(obs_img0)
    allo_obs_buf0 = allo_obs_buf0.at[0].set(grid0)

    def cond_fn(carry):
        rng, stats, timestep, prev_action, prev_reward, h, obs_buf, allo_obs_buf, agent_buf, step = carry
        # Continue while not done with episode (episodes<1) AND under max_steps-1
        # We allow writing index 0..length, so we stop when the *next* write would overflow.
        return jnp.logical_and(stats.episodes < 1, step < max_steps - 1)

    def body_fn(carry):
        rng, stats, timestep, prev_action, prev_reward, h, obs_buf, allo_obs_buf, agent_buf, step = carry

        rng, _rng = jax.random.split(rng)
        # Forward policy on symbolic obs
        dist, _, h = train_state.apply_fn(
            train_state.params,
            {
                "obs_img": timestep.observation["img"][None, None, ...],
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
        obs_buf = obs_buf.at[step + 1].set(timestep_next.observation["img"])
        allo_obs_buf = allo_obs_buf.at[step + 1].set(timestep_next.state.grid)
        pos = timestep_next.state.agent.position        # shape (2,)
        direction = jnp.asarray(timestep_next.state.agent.direction)  # shape ()
        agent_buf = agent_buf.at[step + 1].set(
            jnp.concatenate([pos, direction[None]], axis=0)           # shape (3,)
        )
        return (
            rng,
            stats_next,
            timestep_next,
            action,
            timestep_next.reward,
            h,
            obs_buf,
            allo_obs_buf,
            agent_buf,
            step + 1,
        )

    final = lax.while_loop(
        cond_fn,
        body_fn,
        (rng, stats0, timestep0, prev_action0, prev_reward0, h0, obs_buf0, allo_obs_buf0, agent_buf0, step0),
    )

    _, stats_f, _, _, _, _, obs_buf_f, allo_obs_buf_f, agent_buf_f, step_f = final

    # The actual usable frames run from [0 .. stats_f.length] inclusive of the first frame,
    # but since we wrote one frame per step (including the last post-step obs at index=length),
    # the effective count to keep is `stats_f.length + 1`, capped by max_steps.
    # For downstream convenience, return `length = stats_f.length + 1`.
    length_plus_one = jnp.minimum(step_f + 1, jnp.int32(max_steps))

    return RolloutWithObs(
        stats=stats_f,
        obs_seq=obs_buf_f,
        allo_obs_seq=allo_obs_buf_f,
        agent_seq=agent_buf_f,
        length=length_plus_one,
    )
