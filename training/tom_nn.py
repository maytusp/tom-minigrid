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
from flax.serialization import to_bytes, msgpack_restore, from_state_dict
from flax.core import freeze

from typing import Optional, Dict, Any
import math
from typing import Optional, TypedDict

import distrax

from xminigrid.core.constants import NUM_COLORS, NUM_TILES
from .nn import BatchedRNNModel, ActorCriticInput, EmbeddingEncoder, ActorCriticRNN

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
    predict_frame: bool = False
    predict_action: bool = True

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
        rnn_in = obs_emb
        rnn_out, new_hidden = rnn_core(rnn_in, hidden)  # rnn_out: [B, S, rnn_hidden_dim]

        outputs: Dict[str, Any] = {}

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



class LocalProtagonistRNN(nn.Module):
    num_actions: int
    obs_emb_dim: int = 16
    action_emb_dim: int = 16
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1
    
    # HARDCODED: 128 to match your checkpoint
    head_hidden_dim: int = 128  
    
    img_obs: bool = False
    use_color: bool = True
    direction_obs: bool = False
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs: ActorCriticInput, hidden: jax.Array):
        B, S = inputs["obs_img"].shape[:2]

        # Encoder
        img_encoder = nn.Sequential([
            EmbeddingEncoder(emb_dim=self.obs_emb_dim, use_color=self.use_color),
            nn.Conv(16, (2, 2), padding="VALID", kernel_init=orthogonal(math.sqrt(2)), dtype=self.dtype, param_dtype=self.param_dtype), nn.relu,
            nn.Conv(32, (2, 2), padding="VALID", kernel_init=orthogonal(math.sqrt(2)), dtype=self.dtype, param_dtype=self.param_dtype), nn.relu,
            nn.Conv(64, (2, 2), padding="VALID", kernel_init=orthogonal(math.sqrt(2)), dtype=self.dtype, param_dtype=self.param_dtype), nn.relu,
        ])
        action_encoder = nn.Embed(self.num_actions, self.action_emb_dim)

        # RNN Core
        rnn_core = BatchedRNNModel(self.rnn_hidden_dim, self.rnn_num_layers, dtype=self.dtype, param_dtype=self.param_dtype)
        
        # Heads
        actor = nn.Sequential([
            nn.Dense(self.head_hidden_dim, kernel_init=orthogonal(2), dtype=self.dtype, param_dtype=self.param_dtype),
            nn.tanh,
            nn.Dense(self.num_actions, kernel_init=orthogonal(0.01), dtype=self.dtype, param_dtype=self.param_dtype),
        ])
        critic = nn.Sequential([
            nn.Dense(self.head_hidden_dim, kernel_init=orthogonal(2), dtype=self.dtype, param_dtype=self.param_dtype),
            nn.tanh,
            nn.Dense(1, kernel_init=orthogonal(1.0), dtype=self.dtype, param_dtype=self.param_dtype),
        ])

        # Forward
        obs_emb = img_encoder(inputs["obs_img"].astype(jnp.int32)).reshape(B, S, -1)
        act_emb = action_encoder(inputs["prev_action"])
        
        rnn_input = jnp.concatenate([obs_emb, act_emb, inputs["prev_reward"][..., None]], axis=-1)

        # rnn_seq: [Batch, Seq, Hidden] <- The full sequence we need!
        # new_hidden: [Batch, Layers, Hidden] <- The carry for next step
        rnn_seq, new_hidden = rnn_core(rnn_input, hidden)

        logits = actor(rnn_seq).astype(jnp.float32)
        dist = distrax.Categorical(logits=logits)
        values = critic(rnn_seq)

        # Return rnn_seq as the 4th output so Fusion module can use it
        return dist, jnp.squeeze(values, axis=-1), new_hidden, rnn_seq

    def initialize_carry(self, batch_size):
        return jnp.zeros((batch_size, self.rnn_num_layers, self.rnn_hidden_dim), dtype=self.dtype)

# --- 2. Dual Perspective Model ---
class DualPerspectivePredictor(nn.Module):
    num_actions: int
    fp_emb: int = 16
    fp_rnn: int = 256
    fp_head_dim: int = 128
    tp_emb: int = 16
    tp_rnn: int = 256
    fusion_head_dim: int = 128
    
    # New SR config
    predict_sr: bool = False
    num_states: int = 100
    num_gammas: int = 3
    
    def setup(self):
        # FP Module
        self.fp_module = LocalProtagonistRNN(
            num_actions=self.num_actions,
            obs_emb_dim=self.fp_emb,
            rnn_hidden_dim=self.fp_rnn,
            rnn_num_layers=1,
            head_hidden_dim=self.fp_head_dim,
        )

        # TP Module
        self.tp_encoder = nn.Sequential([
            EmbeddingEncoder(emb_dim=self.tp_emb),
            nn.Conv(16, (2, 2), padding="VALID", kernel_init=orthogonal(math.sqrt(2))), nn.relu,
            nn.Conv(32, (2, 2), padding="VALID", kernel_init=orthogonal(math.sqrt(2))), nn.relu,
            nn.Conv(64, (2, 2), padding="VALID", kernel_init=orthogonal(math.sqrt(2))), nn.relu,
        ])
        self.tp_rnn_core = BatchedRNNModel(self.tp_rnn, num_layers=1)

        # Fusion Head (Action)
        self.fusion_head = nn.Sequential([
            nn.Dense(self.fusion_head_dim, kernel_init=orthogonal(2.0)),
            nn.tanh,
            nn.Dense(self.num_actions, kernel_init=orthogonal(0.01)),
        ])
        
        # SR Head
        if self.predict_sr:
            self.sr_head = nn.Sequential([
                nn.Dense(32, kernel_init=orthogonal(2.0)),
                nn.relu,
                nn.Dense(self.num_states * self.num_gammas, kernel_init=orthogonal(0.01))
            ])

    def __call__(self, inputs_fp, hidden_fp, inputs_tp, hidden_tp):
        _, _, new_h_fp, rnn_seq_fp = self.fp_module(inputs_fp, hidden_fp)
        
        B, S = inputs_tp["obs_img"].shape[:2]
        tp_emb = self.tp_encoder(inputs_tp["obs_img"].astype(jnp.int32)).reshape(B, S, -1)
        rnn_seq_tp, new_h_tp = self.tp_rnn_core(tp_emb, hidden_tp)

        joint_feat = jnp.concatenate([rnn_seq_fp, rnn_seq_tp], axis=-1)
        logits = self.fusion_head(joint_feat)
        
        # Calculate SR if enabled
        pred_sr = None
        if self.predict_sr:
            raw_sr = self.sr_head(joint_feat) # [B, S, States * Gammas]
            reshaped_sr = raw_sr.reshape(B, S, self.num_states, self.num_gammas)
            # Softmax independently over the states axis (axis=2)
            pred_sr = jax.nn.softmax(reshaped_sr, axis=2) 
        
        return logits, pred_sr, new_h_fp, new_h_tp

    def initialize_carry(self, batch_size):
        h_fp = jnp.zeros((batch_size, 1, self.fp_rnn), dtype=jnp.float32)
        h_tp = jnp.zeros((batch_size, 1, self.tp_rnn), dtype=jnp.float32)
        return h_fp, h_tp


# --- 3. Standard Third-Person Model ---

class ThirdPersonPredictor(nn.Module):
    num_actions: int
    tp_emb: int = 16
    tp_rnn: int = 256
    head_dim: int = 128
    
    # New SR config
    predict_sr: bool = False
    num_states: int = 100
    num_gammas: int = 3
    
    def setup(self):
        self.tp_encoder = nn.Sequential([
            EmbeddingEncoder(emb_dim=self.tp_emb),
            nn.Conv(16, (2, 2), padding="VALID", kernel_init=orthogonal(math.sqrt(2))), nn.relu,
            nn.Conv(32, (2, 2), padding="VALID", kernel_init=orthogonal(math.sqrt(2))), nn.relu,
            nn.Conv(64, (2, 2), padding="VALID", kernel_init=orthogonal(math.sqrt(2))), nn.relu,
        ])
        self.tp_rnn_core = BatchedRNNModel(self.tp_rnn, num_layers=1)

        self.head = nn.Sequential([
            nn.Dense(self.head_dim, kernel_init=orthogonal(2.0)),
            nn.tanh,
            nn.Dense(self.num_actions, kernel_init=orthogonal(0.01)),
        ])
        
        # SR Head
        if self.predict_sr:
            self.sr_head = nn.Sequential([
                nn.Dense(32, kernel_init=orthogonal(2.0)),
                nn.relu,
                nn.Dense(self.num_states * self.num_gammas, kernel_init=orthogonal(0.01))
            ])

    def __call__(self, inputs_fp, hidden_fp, inputs_tp, hidden_tp):
        new_h_fp = hidden_fp 
        B, S = inputs_tp["obs_img"].shape[:2]
        tp_emb = self.tp_encoder(inputs_tp["obs_img"].astype(jnp.int32)).reshape(B, S, -1)
        
        rnn_seq_tp, new_h_tp = self.tp_rnn_core(tp_emb, hidden_tp)
        logits = self.head(rnn_seq_tp)
        
        # Calculate SR if enabled
        pred_sr = None
        if self.predict_sr:
            raw_sr = self.sr_head(rnn_seq_tp) 
            reshaped_sr = raw_sr.reshape(B, S, self.num_states, self.num_gammas)
            pred_sr = jax.nn.softmax(reshaped_sr, axis=2)
            
        # NOTE: Returning 4 items now!
        return logits, pred_sr, new_h_fp, new_h_tp

    def initialize_carry(self, batch_size):
        h_fp = jnp.zeros((batch_size, 1, 1), dtype=jnp.float32) 
        h_tp = jnp.zeros((batch_size, 1, self.tp_rnn), dtype=jnp.float32)
        return h_fp, h_tp

def create_model(model_type: str, config: Dict, rng):
    print(f"--- Creating Model: {model_type} ---")
    
    use_sr = config.get('use_sr', False)
    num_states = config.get('num_states', 100)
    
    if model_type == "dual_perspective":
        model = DualPerspectivePredictor(
            num_actions=config['num_actions'],
            fp_emb=config['fp_emb'], 
            fp_rnn=config['fp_rnn'],
            tp_emb=config['tp_emb'], 
            tp_rnn=config['tp_rnn'],
            fusion_head_dim=128,
            predict_sr=use_sr,
            num_states=num_states
        )
    elif model_type == "third_person":
        model = ThirdPersonPredictor(
            num_actions=config['num_actions'],
            tp_emb=config['tp_emb'], 
            tp_rnn=config['tp_rnn'],
            predict_sr=use_sr,
            num_states=num_states
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Init variables
    dummy_fp = {
        "obs_img": jnp.zeros((1, 1, 9, 9, 3), dtype=jnp.int32), 
        "obs_dir": jnp.zeros((1, 1, 4)),
        "prev_action": jnp.zeros((1, 1), dtype=jnp.int32),
        "prev_reward": jnp.zeros((1, 1))
    }
    dummy_tp = {"obs_img": jnp.zeros((1, 1, 9, 9, 2), dtype=jnp.int32)}
    
    h_fp, h_tp = model.initialize_carry(1)
    variables = model.init(rng, dummy_fp, h_fp, dummy_tp, h_tp)
    params = flax.core.unfreeze(variables['params'])

    # Grafting Logic
    if model_type == "dual_perspective" and config.get('p_checkpoint'):
        print(f"Loading FP weights from: {config['p_checkpoint']}")
        with open(config['p_checkpoint'], "rb") as f:
            raw_p = msgpack_restore(f.read())
        
        raw_p_params = raw_p["params"] if "params" in raw_p else raw_p
        
        try:
            params['fp_module'] = from_state_dict(params['fp_module'], raw_p_params)
            print("FP weights grafted successfully.")
        except Exception as e:
            print(f"!!! Grafting Failed: {e}")
            raise e
    
    return model, freeze({'params': params})

# --- 5. Passive Data Utils ---

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