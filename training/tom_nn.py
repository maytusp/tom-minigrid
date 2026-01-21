import jax
import jax.numpy as jnp
import optax


import flax
import flax.linen as nn
from flax import struct
from flax.training.train_state import TrainState
from flax.linen.dtypes import promote_dtype
from flax.linen.initializers import glorot_normal, orthogonal, zeros_init
from flax.typing import Dtype

from typing import Optional, Dict, Any
import math
from typing import Optional, TypedDict

import distrax

from xminigrid.core.constants import NUM_COLORS, NUM_TILES
from .nn import BatchedRNNModel, ActorCriticInput, EmbeddingEncoder

TOTAL_TILES = NUM_TILES + 4
class AuxiliaryPredictorRNN(nn.Module):
    #TODO Add the version that can get action and direction of an agent as inputs
    """
    Same encoders and RNN core as ActorCriticRNN, but with heads for:
      - next-frame prediction: per-cell Categorical over TOTAL_TILES
      - other-agent next-action prediction: Categorical over num_actions
    You can turn either head on or off via `predict_frame` / `predict_action`.

    Returns:
      outputs: Dict with (present if enabled)
        - "action_dist": distrax.Categorical over other agent actions, shape [B, S, num_actions]
        - "frame_logits": jnp.float32 logits with shape [B, S, view_size, view_size, TOTAL_TILES]
      new_hidden: jnp.ndarray, the new RNN hidden state (same shape as ActorCriticRNN)
    """
    num_actions: int
    view_size: int
    predict_frame: bool = True
    predict_action: bool = False

    # encoder/core config
    obs_emb_dim: int = 16
    action_emb_dim: int = 16
    rnn_hidden_dim: int = 64
    rnn_num_layers: int = 1
    head_hidden_dim: int = 64
    use_dir: bool = False # use agent direction as input
    use_action: bool = False # use agent action as input
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
        if self.use_dir:
            direction_encoder = nn.Dense(self.action_emb_dim, dtype=self.dtype, param_dtype=self.param_dtype)
       
        if self.use_action:
            action_encoder = nn.Embed(self.num_actions, self.action_emb_dim)

        rnn_core = BatchedRNNModel(
            self.rnn_hidden_dim, self.rnn_num_layers, dtype=self.dtype, param_dtype=self.param_dtype
        )

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
        # We keep it simple with MLP -> Dense to view_size*view_size*TOTAL_TILES and reshape.
        if self.predict_frame:
            frame_head = nn.Sequential(
                [
                    nn.Dense(self.head_hidden_dim, kernel_init=orthogonal(2.0),
                             dtype=self.dtype, param_dtype=self.param_dtype),
                    nn.tanh,
                    nn.Dense(self.view_size * self.view_size * TOTAL_TILES,
                             kernel_init=orthogonal(0.01),
                             dtype=self.dtype, param_dtype=self.param_dtype),
                ]
            )

        # === Build sequence inputs (identical concat) ===
        # [B, S, ...] -> flatten spatial after conv/emb
        obs_emb = img_encoder(inputs["obs_img"].astype(jnp.int32)).reshape(B, S, -1)

        if self.use_action:
            act_emb = action_encoder(inputs["prev_action"])

        # Concatenate: [B, S, hidden_dim + 2*action_emb_dim + 1]
        # rnn_in = jnp.concatenate([obs_emb, dir_emb, act_emb, inputs["prev_reward"][..., None]], axis=-1)
        rnn_in = obs_emb

        # === Core ===
        rnn_out, new_hidden = rnn_core(rnn_in, hidden)  # rnn_out: [B, S, rnn_hidden_dim]

        outputs: Dict[str, Any] = {}

        # === Heads forward ===
        if self.predict_action:
            # Cast to full precision for stable softmax
            logits = other_actor_head(rnn_out).astype(jnp.float32)  # [B, S, num_actions]
            outputs["action_dist"] = distrax.Categorical(logits=logits)

        if self.predict_frame:
            raw = frame_head(rnn_out).astype(jnp.float32)  # [B, S, V*V*TOTAL_TILES]
            frame_logits = raw.reshape(B, S, self.view_size, self.view_size, TOTAL_TILES)
            outputs["frame_logits"] = frame_logits  # Per-cell tile logits

        return outputs, new_hidden

    def initialize_carry(self, batch_size):
        return jnp.zeros((batch_size, self.rnn_num_layers, self.rnn_hidden_dim), dtype=self.dtype)

class PassiveTargets(struct.PyTreeNode):
    """
    next_frame: [B, S, V, V] int32   (tile ids in [0, TOTAL_TILES))
    next_action: [B, S] int32        (other agent action ids)
    mask: [B, S] float32             (1.0 for valid steps, 0.0 for padding/terminal)
    spatial_mask: [B, S, V, V] float32 (change detection mask)
    """
    next_frame: Optional[jax.Array] = None
    next_action: Optional[jax.Array] = None
    mask: Optional[jax.Array] = None
    spatial_mask: Optional[jax.Array] = None

def passive_update(
    train_state: TrainState,
    init_hstate: jax.Array,
    *,
    inputs: Dict[str, jax.Array],
    targets: PassiveTargets,
    view_size: int,
    predict_frame: bool = True,
    predict_action: bool = False,
    frame_weight: float = 1.0,
    action_weight: float = 1.0,
    static_pixel_weight: float = 0.1, 
):
    B, S = inputs["obs_img"].shape[:2]
    V = view_size
    seq_mask = targets.mask if targets.mask is not None else jnp.ones((B, S), jnp.float32)

    def _loss_fn(params):
        outputs, _ = train_state.apply_fn({'params': params}, inputs, init_hstate)
        total_loss = 0.0
        aux = {}

        if predict_frame:
            frame_logits = outputs["frame_logits"].astype(jnp.float32)
            frame_labels = targets.next_frame.astype(jnp.int32)

            # Flatten
            logits_flat = frame_logits.reshape(B, S, V * V, frame_logits.shape[-1])
            labels_flat = frame_labels.reshape(B, S, V * V)

            # 1. Cross Entropy per cell
            ce_cells = optax.softmax_cross_entropy_with_integer_labels(logits_flat, labels_flat)

            # 2. Calculate Spatial Weights
            # Start with the sequence mask (B, S, 1)
            valid_step_mask = seq_mask[..., None] 

            if targets.spatial_mask is not None:
                # targets.spatial_mask is 1.0 for CHANGE, 0.0 for STATIC
                change_mask_flat = targets.spatial_mask.reshape(B, S, V * V)
                
                # Apply the mixing formula:
                # If Change: 1.0 * 1.0 = 1.0
                # If Static: 1.0 * static_pixel_weight
                spatial_weights = change_mask_flat + (1.0 - change_mask_flat) * static_pixel_weight
                
                # Combine sequence validity with spatial weights
                final_mask = valid_step_mask * spatial_weights
            else:
                final_mask = valid_step_mask

            # 3. Apply Weighted Mask
            ce_weighted = ce_cells * final_mask

            loss_dynamic_term = (ce_cells * change_mask_flat * valid_step_mask).sum() / (change_mask_flat.sum())
            loss_static_term = (ce_cells * (1-change_mask_flat) * valid_step_mask).sum() / ((1-change_mask_flat).sum())

            # 4. Normalize
            # Important: Normalize by the sum of weights, not just count of pixels
            # This ensures the gradient magnitude stays stable regardless of scene activity
            denom = jnp.maximum(final_mask.sum(), 1.0)
            frame_loss = ce_weighted.sum() / denom

            aux["frame_loss"] = frame_loss
            aux["frame_loss_dynamic"] = loss_dynamic_term
            aux["frame_loss_static"] = loss_static_term
            total_loss = total_loss + frame_weight * frame_loss

        # ... (Action loss remains the same) ...
        if predict_action:
            dist = outputs["action_dist"]
            nll = -dist.log_prob(targets.next_action)
            nll_masked = nll * seq_mask
            denom = jnp.maximum(seq_mask.sum(), 1.0)
            action_loss = nll_masked.sum() / denom
            aux["action_loss"] = action_loss
            total_loss = total_loss + action_weight * action_loss

        return total_loss, aux

    (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)
    logs = {"total_loss": loss}
    logs.update(aux)
    return train_state, logs


def build_passive_batch_from_sequences(
    *,
    obs_seq: jax.Array,          # [B,S, H, W, 2] or your packed symbolic obs
    prev_action_seq: jax.Array,  # [B,S] (your own previous action)
    prev_reward_seq: jax.Array,  # [B,S]
    next_frame_seq: Optional[jax.Array],   # [B,S, V, V] (tile ids)  OR None
    next_other_action_seq: Optional[jax.Array],  # [B,S]            OR None
    done_seq: jax.Array,         # [B,S] 1.0 when episode ended *at* t (i.e., obs_{t+1} is invalid)
    spatial_mask_seq: Optional[jax.Array] = None,
):
    """
    Prepares teacher-forcing inputs at time t and supervision at t+1.
    Assumes last step has no valid target, so mask it out.
    """
    B, S = obs_seq.shape[:2]

    # Inputs are taken at t = 0..S-2; we keep shapes [B,S,...] by shifting targets and masking the last step.
    inputs = {
        "obs_img": obs_seq,              # [B,S,...] as your model expects
        "prev_action": prev_action_seq,
        "prev_reward": prev_reward_seq,
    }

    # Valid if not terminal *and* not on the very last time index
    # If your `done` indicates episode finished *after* taking a_t (i.e., obs_{t+1} is terminal),
    # then targets at t are invalid whenever done[t] == 1.
    valid_steps = (1.0 - done_seq).astype(jnp.float32)
    # Optionally mask the final index to be safe (no t+1 exists in padded batches)
    valid_steps = valid_steps.at[:, -1].set(0.0)

    targets = PassiveTargets(
        next_frame=next_frame_seq,
        next_action=next_other_action_seq,
        mask=valid_steps,
        spatial_mask=spatial_mask_seq,
    )

    return inputs, targets