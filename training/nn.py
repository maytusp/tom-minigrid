# Model adapted from minigrid baselines:
# https://github.com/lcswillems/rl-starter-files/blob/master/model.py
import math
from typing import Optional, TypedDict

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.dtypes import promote_dtype
from flax.linen.initializers import glorot_normal, orthogonal, zeros_init
from flax.typing import Dtype

from xminigrid.core.constants import NUM_COLORS, NUM_TILES


class GRU(nn.Module):
    hidden_dim: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, xs, init_state):
        seq_len, input_dim = xs.shape
        # this init might not be optimal, for example bias for reset gate should be -1 (for now ok)
        Wi = self.param("Wi", glorot_normal(in_axis=1, out_axis=0), (self.hidden_dim * 3, input_dim), self.param_dtype)
        Wh = self.param("Wh", orthogonal(column_axis=0), (self.hidden_dim * 3, self.hidden_dim), self.param_dtype)
        bi = self.param("bi", zeros_init(), (self.hidden_dim * 3,), self.param_dtype)
        bn = self.param("bn", zeros_init(), (self.hidden_dim,), self.param_dtype)

        def _step_fn(h, x):
            igates = jnp.split(Wi @ x + bi, 3)
            hgates = jnp.split(Wh @ h, 3)

            reset = nn.sigmoid(igates[0] + hgates[0])
            update = nn.sigmoid(igates[1] + hgates[1])
            new = nn.tanh(igates[2] + reset * (hgates[2] + bn))
            next_h = (1 - update) * new + update * h

            return next_h, next_h

        # cast to the computation dtype
        xs, init_state, Wi, Wh, bi, bn = promote_dtype(xs, init_state, Wi, Wh, bi, bn, dtype=self.dtype)

        last_state, all_states = jax.lax.scan(_step_fn, init=init_state, xs=xs)
        return all_states, last_state


class RNNModel(nn.Module):
    hidden_dim: int
    num_layers: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, xs, init_state):
        # xs: [seq_len, input_dim]
        # init_state: [num_layers, hidden_dim]
        outs, states = [], []
        for layer in range(self.num_layers):
            xs, state = GRU(self.hidden_dim, self.dtype, self.param_dtype)(xs, init_state[layer])
            outs.append(xs)
            states.append(state)

        # sum outputs from all layers, kinda like in ResNet
        return jnp.array(outs).sum(0), jnp.array(states)


BatchedRNNModel = flax.linen.vmap(
    RNNModel, variable_axes={"params": None}, split_rngs={"params": False}, axis_name="batch"
)


class EmbeddingEncoder(nn.Module):
    emb_dim: int = 16
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, img):
        entity_emb = nn.Embed(NUM_TILES, self.emb_dim, self.dtype, self.param_dtype)
        color_emb = nn.Embed(NUM_COLORS, self.emb_dim, self.dtype, self.param_dtype)

        # [..., channels]
        img_emb = jnp.concatenate(
            [
                entity_emb(img[..., 0]),
                color_emb(img[..., 1]),
            ],
            axis=-1,
        )
        return img_emb


class ActorCriticInput(TypedDict):
    obs_img: jax.Array
    obs_dir: jax.Array
    prev_action: jax.Array
    prev_reward: jax.Array


class ActorCriticRNN(nn.Module):
    num_actions: int
    obs_emb_dim: int = 16
    action_emb_dim: int = 16
    rnn_hidden_dim: int = 64
    rnn_num_layers: int = 1
    head_hidden_dim: int = 64
    img_obs: bool = False
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs: ActorCriticInput, hidden: jax.Array) -> tuple[distrax.Categorical, jax.Array, jax.Array]:
        B, S = inputs["obs_img"].shape[:2]

        # encoder from https://github.com/lcswillems/rl-starter-files/blob/master/model.py
        if self.img_obs:
            img_encoder = nn.Sequential(
                [
                    nn.Conv(
                        16,
                        (3, 3),
                        strides=2,
                        padding="VALID",
                        kernel_init=orthogonal(math.sqrt(2)),
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                    ),
                    nn.relu,
                    nn.Conv(
                        32,
                        (3, 3),
                        strides=2,
                        padding="VALID",
                        kernel_init=orthogonal(math.sqrt(2)),
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                    ),
                    nn.relu,
                    nn.Conv(
                        32,
                        (3, 3),
                        strides=2,
                        padding="VALID",
                        kernel_init=orthogonal(math.sqrt(2)),
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                    ),
                    nn.relu,
                    nn.Conv(
                        32,
                        (3, 3),
                        strides=2,
                        padding="VALID",
                        kernel_init=orthogonal(math.sqrt(2)),
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                    ),
                ]
            )
        else:
            img_encoder = nn.Sequential(
                [
                    # For small dims nn.Embed is extremely slow in bf16, so we leave everything in default dtypes
                    EmbeddingEncoder(emb_dim=self.obs_emb_dim),
                    nn.Conv(
                        16,
                        (2, 2),
                        padding="VALID",
                        kernel_init=orthogonal(math.sqrt(2)),
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                    ),
                    nn.relu,
                    nn.Conv(
                        32,
                        (2, 2),
                        padding="VALID",
                        kernel_init=orthogonal(math.sqrt(2)),
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                    ),
                    nn.relu,
                    nn.Conv(
                        64,
                        (2, 2),
                        padding="VALID",
                        kernel_init=orthogonal(math.sqrt(2)),
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                    ),
                    nn.relu,
                ]
            )
        action_encoder = nn.Embed(self.num_actions, self.action_emb_dim)
        direction_encoder = nn.Dense(self.action_emb_dim, dtype=self.dtype, param_dtype=self.param_dtype)

        rnn_core = BatchedRNNModel(
            self.rnn_hidden_dim, self.rnn_num_layers, dtype=self.dtype, param_dtype=self.param_dtype
        )
        actor = nn.Sequential(
            [
                nn.Dense(
                    self.head_hidden_dim, kernel_init=orthogonal(2), dtype=self.dtype, param_dtype=self.param_dtype
                ),
                nn.tanh,
                nn.Dense(
                    self.num_actions, kernel_init=orthogonal(0.01), dtype=self.dtype, param_dtype=self.param_dtype
                ),
            ]
        )
        critic = nn.Sequential(
            [
                nn.Dense(
                    self.head_hidden_dim, kernel_init=orthogonal(2), dtype=self.dtype, param_dtype=self.param_dtype
                ),
                nn.tanh,
                nn.Dense(1, kernel_init=orthogonal(1.0), dtype=self.dtype, param_dtype=self.param_dtype),
            ]
        )

        # [batch_size, seq_len, ...]
        obs_emb = img_encoder(inputs["obs_img"].astype(jnp.int32)).reshape(B, S, -1)
        dir_emb = direction_encoder(inputs["obs_dir"])
        act_emb = action_encoder(inputs["prev_action"])

        # [batch_size, seq_len, hidden_dim + 2 * act_emb_dim + 1]
        out = jnp.concatenate([obs_emb, dir_emb, act_emb, inputs["prev_reward"][..., None]], axis=-1)

        # core networks
        out, new_hidden = rnn_core(out, hidden)

        # casting to full precision for the loss, as softmax/log_softmax
        # (inside Categorical) is not stable in bf16
        logits = actor(out).astype(jnp.float32)

        dist = distrax.Categorical(logits=logits)
        values = critic(out)

        return dist, jnp.squeeze(values, axis=-1), new_hidden

    def initialize_carry(self, batch_size):
        return jnp.zeros((batch_size, self.rnn_num_layers, self.rnn_hidden_dim), dtype=self.dtype)




class AuxiliaryPredictorRNN(nn.Module):
    """
    Same encoders and RNN core as ActorCriticRNN, but with heads for:
      - next-frame prediction: per-cell Categorical over NUM_TILES
      - other-agent next-action prediction: Categorical over num_actions
    You can turn either head on or off via `predict_frame` / `predict_action`.

    Returns:
      outputs: Dict with (present if enabled)
        - "action_dist": distrax.Categorical over other agent actions, shape [B, S, num_actions]
        - "frame_logits": jnp.float32 logits with shape [B, S, view_size, view_size, NUM_TILES]
      new_hidden: jnp.ndarray, the new RNN hidden state (same shape as ActorCriticRNN)
    """
    num_actions: int
    view_size: int
    predict_frame: bool = True
    predict_action: bool = True

    # encoder/core config
    obs_emb_dim: int = 16
    action_emb_dim: int = 16
    rnn_hidden_dim: int = 64
    rnn_num_layers: int = 1
    head_hidden_dim: int = 64
    img_obs: bool = False

    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        inputs: ActorCriticInput,
        hidden: jax.Array,
    ):
        """
        Args:
          inputs: same TypedDict as ActorCriticRNN (obs_img, obs_dir, prev_action, prev_reward)
          hidden: [batch, rnn_num_layers, rnn_hidden_dim]
        Returns:
          (outputs, new_hidden)
        """
        B, S = inputs["obs_img"].shape[:2]

        # === Encoders (identical to ActorCriticRNN) ===
        if self.img_obs:
            img_encoder = nn.Sequential(
                [
                    nn.Conv(16, (3, 3), strides=2, padding="VALID",
                            kernel_init=orthogonal(math.sqrt(2)),
                            dtype=self.dtype, param_dtype=self.param_dtype),
                    nn.relu,
                    nn.Conv(32, (3, 3), strides=2, padding="VALID",
                            kernel_init=orthogonal(math.sqrt(2)),
                            dtype=self.dtype, param_dtype=self.param_dtype),
                    nn.relu,
                    nn.Conv(32, (3, 3), strides=2, padding="VALID",
                            kernel_init=orthogonal(math.sqrt(2)),
                            dtype=self.dtype, param_dtype=self.param_dtype),
                    nn.relu,
                    nn.Conv(32, (3, 3), strides=2, padding="VALID",
                            kernel_init=orthogonal(math.sqrt(2)),
                            dtype=self.dtype, param_dtype=self.param_dtype),
                ]
            )
        else:
            img_encoder = nn.Sequential(
                [
                    EmbeddingEncoder(emb_dim=self.obs_emb_dim, dtype=self.dtype, param_dtype=self.param_dtype),
                    nn.Conv(16, (2, 2), padding="VALID",
                            kernel_init=orthogonal(math.sqrt(2)),
                            dtype=self.dtype, param_dtype=self.param_dtype),
                    nn.relu,
                    nn.Conv(32, (2, 2), padding="VALID",
                            kernel_init=orthogonal(math.sqrt(2)),
                            dtype=self.dtype, param_dtype=self.param_dtype),
                    nn.relu,
                    nn.Conv(64, (2, 2), padding="VALID",
                            kernel_init=orthogonal(math.sqrt(2)),
                            dtype=self.dtype, param_dtype=self.param_dtype),
                    nn.relu,
                ]
            )

        action_encoder = nn.Embed(self.num_actions, self.action_emb_dim)
        direction_encoder = nn.Dense(self.action_emb_dim, dtype=self.dtype, param_dtype=self.param_dtype)

        rnn_core = BatchedRNNModel(
            self.rnn_hidden_dim, self.rnn_num_layers, dtype=self.dtype, param_dtype=self.param_dtype
        )

        # === Heads (new) ===
        # Other-agent action head (same shape/init style as your actor).
        if self.predict_action:
            other_actor_head = nn.Sequential(
                [
                    nn.Dense(self.head_hidden_dim, kernel_init=orthogonal(2.0),
                             dtype=self.dtype, param_dtype=self.param_dtype),
                    nn.tanh,
                    nn.Dense(self.num_actions, kernel_init=orthogonal(0.01),
                             dtype=self.dtype, param_dtype=self.param_dtype),
                ]
            )

        # Next-frame head: decode RNN features to per-cell tile logits.
        # We keep it simple with MLP -> Dense to view_size*view_size*NUM_TILES and reshape.
        if self.predict_frame:
            frame_head = nn.Sequential(
                [
                    nn.Dense(self.head_hidden_dim, kernel_init=orthogonal(2.0),
                             dtype=self.dtype, param_dtype=self.param_dtype),
                    nn.tanh,
                    nn.Dense(self.view_size * self.view_size * NUM_TILES,
                             kernel_init=orthogonal(0.01),
                             dtype=self.dtype, param_dtype=self.param_dtype),
                ]
            )

        # === Build sequence inputs (identical concat) ===
        # [B, S, ...] -> flatten spatial after conv/emb
        obs_emb = img_encoder(inputs["obs_img"].astype(jnp.int32)).reshape(B, S, -1)
        dir_emb = direction_encoder(inputs["obs_dir"])
        act_emb = action_encoder(inputs["prev_action"])

        # Concatenate: [B, S, hidden_dim + 2*action_emb_dim + 1]
        rnn_in = jnp.concatenate([obs_emb, dir_emb, act_emb, inputs["prev_reward"][..., None]], axis=-1)

        # === Core ===
        rnn_out, new_hidden = rnn_core(rnn_in, hidden)  # rnn_out: [B, S, rnn_hidden_dim]

        outputs: Dict[str, Any] = {}

        # === Heads forward ===
        if self.predict_action:
            # Cast to full precision for stable softmax
            logits = other_actor_head(rnn_out).astype(jnp.float32)  # [B, S, num_actions]
            outputs["action_dist"] = distrax.Categorical(logits=logits)

        if self.predict_frame:
            raw = frame_head(rnn_out).astype(jnp.float32)  # [B, S, V*V*NUM_TILES]
            frame_logits = raw.reshape(B, S, self.view_size, self.view_size, NUM_TILES)
            outputs["frame_logits"] = frame_logits  # Per-cell tile logits

        return outputs, new_hidden

    def initialize_carry(self, batch_size):
        return jnp.zeros((batch_size, self.rnn_num_layers, self.rnn_hidden_dim), dtype=self.dtype)