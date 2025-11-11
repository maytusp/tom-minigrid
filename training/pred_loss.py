import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from typing import Optional, Dict, Any

class PassiveTargets(struct.PyTreeNode):
    """
    next_frame: [B, S, V, V] int32   (tile ids in [0, NUM_TILES))
    next_action: [B, S] int32        (other agent action ids)
    mask: [B, S] float32             (1.0 for valid steps, 0.0 for padding/terminal)
    """
    next_frame: Optional[jax.Array] = None
    next_action: Optional[jax.Array] = None
    mask: Optional[jax.Array] = None


def passive_update(
    train_state: TrainState,
    init_hstate: jax.Array,
    *,
    inputs: Dict[str, jax.Array],       # keys: "obs_img", "obs_dir", "prev_action", "prev_reward"; shapes [B,S,...]
    targets: PassiveTargets,            # see struct above
    view_size: int,
    predict_frame: bool = True,
    predict_action: bool = True,
    frame_weight: float = 1.0,
    action_weight: float = 1.0,
):
    """
    One optimization step for the passive predictor.

    The model must be AuxiliaryPredictorRNN or any module that returns:
        outputs, new_hidden = apply_fn(params, inputs, init_hstate)
        where outputs may contain:
          - "frame_logits": [B,S,V,V,NUM_TILES]
          - "action_dist": distrax.Categorical over [B,S,num_actions]
    """

    B, S = inputs["obs_img"].shape[:2]
    V = view_size
    mask = targets.mask if targets.mask is not None else jnp.ones((B, S), jnp.float32)

    def _loss_fn(params):
        outputs, _ = train_state.apply_fn(params, inputs, init_hstate)

        total_loss = 0.0
        aux = {}

        # ---- Next-frame loss (per-cell CE over NUM_TILES) ----
        if predict_frame:
            # logits: [B,S,V,V,NUM_TILES], labels: [B,S,V,V]
            frame_logits = outputs["frame_logits"].astype(jnp.float32)
            frame_labels = targets.next_frame.astype(jnp.int32)

            # Flatten cells but keep [B,S] to apply a per-step mask
            logits_flat = frame_logits.reshape(B, S, V * V, frame_logits.shape[-1])  # [B,S, VV, C]
            labels_flat = frame_labels.reshape(B, S, V * V)                          # [B,S, VV]

            # Cross-entropy per cell
            ce_cells = optax.softmax_cross_entropy_with_integer_labels(
                logits_flat, labels_flat
            )  # [B,S, VV]

            # Mask per step (broadcast to cells), then average only over valid cells
            step_mask = mask[..., None]                       # [B,S,1]
            ce_masked = ce_cells * step_mask                  # [B,S,VV]
            denom = jnp.maximum(step_mask.sum() * (V * V), 1.0)
            frame_loss = ce_masked.sum() / denom

            aux["frame_loss"] = frame_loss
            total_loss = total_loss + frame_weight * frame_loss

        # ---- Next-action loss (CE over num_actions) ----
        if predict_action:
            dist = outputs["action_dist"]                     # distrax.Categorical
            # Negative log-likelihood of the true next action
            nll = -dist.log_prob(targets.next_action)         # [B,S]
            nll_masked = nll * mask
            denom = jnp.maximum(mask.sum(), 1.0)
            action_loss = nll_masked.sum() / denom

            aux["action_loss"] = action_loss
            total_loss = total_loss + action_weight * action_loss

        return total_loss, aux

    (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params)
    (loss, grads) = jax.lax.pmean((loss, grads), axis_name="devices")
    train_state = train_state.apply_gradients(grads=grads)

    # For logging across devices
    logs = {"total_loss": loss}
    if "frame_loss" in aux:
        logs["frame_loss"] = jax.lax.pmean(aux["frame_loss"], axis_name="devices")
    if "action_loss" in aux:
        logs["action_loss"] = jax.lax.pmean(aux["action_loss"], axis_name="devices")

    return train_state, logs


def build_passive_batch_from_sequences(
    *,
    obs_seq: jax.Array,          # [B,S, H, W, 2] or your packed symbolic obs
    dir_seq: jax.Array,          # [B,S, dir_dim]
    prev_action_seq: jax.Array,  # [B,S] (your own previous action)
    prev_reward_seq: jax.Array,  # [B,S]
    next_frame_seq: Optional[jax.Array],   # [B,S, V, V] (tile ids)  OR None
    next_other_action_seq: Optional[jax.Array],  # [B,S]            OR None
    done_seq: jax.Array,         # [B,S] 1.0 when episode ended *at* t (i.e., obs_{t+1} is invalid)
):
    """
    Prepares teacher-forcing inputs at time t and supervision at t+1.
    Assumes last step has no valid target, so mask it out.
    """
    B, S = obs_seq.shape[:2]

    # Inputs are taken at t = 0..S-2; we keep shapes [B,S,...] by shifting targets and masking the last step.
    inputs = {
        "obs_img": obs_seq,              # [B,S,...] as your model expects
        "obs_dir": dir_seq,
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
    )

    return inputs, targets
