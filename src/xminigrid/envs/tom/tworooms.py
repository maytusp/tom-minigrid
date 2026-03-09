# Sally-Anne-Style Test (with STAR disappearing on reach) — JAX-safe, (y,x) consistent

from __future__ import annotations

import dataclasses
import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple, Any

from flax import struct  # <-- make carry a JAX pytree

from ...core.constants import TILES_REGISTRY, Colors, Tiles, NUM_LAYERS
from ...core.goals import AgentOnTileGoal, check_goal
from ...core.grid import empty_world, sample_coordinates, sample_direction, horizontal_line, vertical_line, rectangle
from ...core.rules import EmptyRule, check_rule
from ...core.actions import take_action
from ...core.observation import minigrid_field_of_view as transparent_field_of_view
from ...core.observation import get_agent_layer
from ...environment import Environment, EnvParams
from ...types import AgentState, State, TimeStep, StepType, IntOrArray

# goals and rules are hardcoded for minigrid envs
_goal_encoding = AgentOnTileGoal(tile=TILES_REGISTRY[Tiles.GOAL, Colors.GREEN]).encode()
_rule_encoding = EmptyRule().encode()[None, ...]


# --- Carry for this environment (stores bookkeeping for Sally-Anne test) ---
@struct.dataclass
class SwapCarry:
    star_reached: jnp.ndarray            # bool[] scalar
    star_reached_step: jnp.ndarray       # int32[] scalar (Records T_m)
    swap_done: jnp.ndarray               # bool[] scalar
    goal_yx: jnp.ndarray                 # int32[2]
    star_yx: jnp.ndarray                 # int32[2]
    empty_tile: jnp.ndarray              # tile dtype
    p_swap_test: jnp.ndarray             # float32[] scalar
    doors_yx: jnp.ndarray                # int32[2,2] -> [left_door_yx, right_door_yx]
    door_close_delay: jnp.ndarray        # int32[] scalar (The actual delay for this episode)
    
class ToMEnvParams(EnvParams):
    testing: bool = struct.field(pytree_node=False, default=True)
    swap_prob: float = struct.field(pytree_node=False, default=1.0)
    use_color: bool = struct.field(pytree_node=False, default=True)
    
    # If random_door_close_delay is True: delay is sampled from [1, door_close_delay]
    # If random_door_close_delay is False: delay is exactly door_close_delay
    door_close_delay: int = struct.field(pytree_node=False, default=1)
    random_door_close_delay: bool = struct.field(pytree_node=False, default=False)

class TwoRooms(Environment[EnvParams, SwapCarry]):
    """Four squares, one goal, one star.
    Task: go to STAR first (adjacent & facing) → STAR disappears. Then go to GOAL.
    Test-time Sally–Anne: with prob `swap_prob` right after STAR is reached, swap the GOAL with a
    randomly chosen SQUARE.
    
    Doors close logic:
    - If random_door_close_delay=False: Closes `door_close_delay` steps after star reach.
    - If random_door_close_delay=True: Closes random(1, door_close_delay) steps after star reach.
    """
    def num_actions(self, params: EnvParamsT) -> int:
        return 6

    def default_params(self, **kwargs) -> ToMEnvParams:
        params = ToMEnvParams(height=13, width=13)
        params = params.replace(**{k: v for k, v in kwargs.items() if k in {
            "height","width","view_size","max_steps","render_mode",
            "testing","swap_prob", "door_close_delay", "random_door_close_delay"}})
        if params.max_steps is None:
            params = params.replace(max_steps=4 * (params.height * params.width))
        return params


    @staticmethod
    def _sample_row_from_mask(key: jax.Array, rows: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        idxs = jnp.nonzero(mask, size=rows.shape[0], fill_value=0)[0]    # [N] padded with 0s
        count = jnp.sum(mask).astype(jnp.int32)
        count = jnp.maximum(count, 1)  # avoid 0; if none valid, falls back to idx 0 (safe due to padding)
        ridx = jax.random.randint(key, shape=(), minval=0, maxval=count)
        sel = idxs[ridx]
        return rows[sel]

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _is_adjacent_and_facing(agent: AgentState, target_yx: jnp.ndarray) -> jnp.ndarray:
        """True iff agent is adjacent to target and facing it.
        Directions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT. Positions are (y, x)."""
        ay = agent.position[0]
        ax = agent.position[1]
        ty = target_yx[0]
        tx = target_yx[1]
        dy = ty - ay
        dx = tx - ax

        adjacent = (jnp.abs(dy) + jnp.abs(dx)) == 1

        facing = jnp.logical_or(
            jnp.logical_or(
                jnp.logical_and(agent.direction == 0, jnp.logical_and(dy == -1, dx == 0)),  # UP
                jnp.logical_and(agent.direction == 1, jnp.logical_and(dy ==  0, dx == 1)),  # RIGHT
            ),
            jnp.logical_or(
                jnp.logical_and(agent.direction == 2, jnp.logical_and(dy ==  1, dx == 0)),  # DOWN
                jnp.logical_and(agent.direction == 3, jnp.logical_and(dy ==  0, dx == -1)), # LEFT
            ),
        )
        return jnp.logical_and(adjacent, facing)

    @staticmethod
    def _place(grid: jnp.ndarray, yx: jnp.ndarray, tile) -> jnp.ndarray:
        """Place tile at (y, x) using JAX-safe scalar indices."""
        y = jnp.asarray(yx[0], jnp.int32)
        x = jnp.asarray(yx[1], jnp.int32)
        return grid.at[y, x].set(tile)

    @staticmethod
    def _get_building_bounds(H: int, W: int):
        """
        Calculates the walls of the two-room building 
        Returns: y0 (top), y1 (bot), x0 (left), x1 (right), midx (split)
        """
        h_room = H // 2 + 1
        w_room = W // 2 + 1
        
        y0 = (H - h_room) // 2 - 1
        y1 = y0 + h_room + 1
        x0 = (W - w_room) // 2 - 1
        x1 = x0 + w_room + 1
        
        midx = (x0 + x1) // 2
        return y0, y1, x0, x1, midx

    # -------------------------
    # Problem generation
    # -------------------------
    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State[SwapCarry]:
        
        H, W = params.height, params.width
        grid = empty_world(H, W)
        empty_tile = grid[1, 1]

        WALL = TILES_REGISTRY[Tiles.WALL, Colors.GREY]
        DOOR = TILES_REGISTRY[Tiles.DOOR_OPEN, Colors.GREEN] 

        y0, y1, x0, x1, midx = self._get_building_bounds(H, W)

        # 7x7 ring + vertical separator
        grid = rectangle(grid, x0, y0, h=(y1 - y0 + 1), w=(x1 - x0 + 1), tile=WALL)
        grid = vertical_line(grid, x=midx, y=y0 + 1, length=(y1 - y0 - 1), tile=WALL)

        # Doors: one for each room, on the OUTER ring
        key, k_left, k_right, k_pick = jax.random.split(key, 4)

        left_top_xs   = jnp.arange(x0 + 1, midx, dtype=jnp.int32)
        left_bot_xs   = jnp.arange(x0 + 1, midx, dtype=jnp.int32)
        left_edge_ys  = jnp.arange(y0 + 1, y1, dtype=jnp.int32)
        left_candidates = jnp.concatenate([
            jnp.stack([jnp.full_like(left_top_xs,  y0), left_top_xs], axis=1),
            jnp.stack([jnp.full_like(left_bot_xs,  y1), left_bot_xs], axis=1),
            jnp.stack([left_edge_ys, jnp.full_like(left_edge_ys, x0)], axis=1),
        ], axis=0)

        right_top_xs  = jnp.arange(midx + 1, x1, dtype=jnp.int32)
        right_bot_xs  = jnp.arange(midx + 1, x1, dtype=jnp.int32)
        right_edge_ys = jnp.arange(y0 + 1, y1, dtype=jnp.int32)
        right_candidates = jnp.concatenate([
            jnp.stack([jnp.full_like(right_top_xs, y0), right_top_xs], axis=1),
            jnp.stack([jnp.full_like(right_bot_xs, y1), right_bot_xs], axis=1),
            jnp.stack([right_edge_ys, jnp.full_like(right_edge_ys, x1)], axis=1),
        ], axis=0)

        li = jax.random.randint(k_left,  shape=(), minval=0, maxval=left_candidates.shape[0])
        ri = jax.random.randint(k_right, shape=(), minval=0, maxval=right_candidates.shape[0])
        left_door_yx  = left_candidates[li]
        right_door_yx = right_candidates[ri]
        grid = grid.at[left_door_yx[0],  left_door_yx[1]].set(DOOR)
        grid = grid.at[right_door_yx[0], right_door_yx[1]].set(DOOR)

        # Room Masks
        Y = jnp.arange(H, dtype=jnp.int32)[:, None]
        X = jnp.arange(W, dtype=jnp.int32)[None, :]

        left_room_mask  = (Y > y0) & (Y < y1) & (X > x0) & (X <  midx)
        right_room_mask = (Y > y0) & (Y < y1) & (X > midx) & (X <  x1)
        outside_mask    = jnp.logical_not(left_room_mask | right_room_mask)

        # Goal & Star
        goal_in_left = jax.random.bernoulli(k_pick, p=jnp.asarray(0.5, jnp.float32))
        key, k_goal, k_star, k_agent = jax.random.split(key, 4)

        goal_mask = jax.lax.select(goal_in_left, left_room_mask, right_room_mask)
        goal_yx = sample_coordinates(k_goal, grid, num=1, mask=goal_mask)[0]
        grid = self._place(grid, goal_yx, TILES_REGISTRY[Tiles.GOAL, Colors.BLUE])

        corner_candidates = jnp.array([
            [0, 0],                # Top-Left
            [0, W - 1],            # Top-Right
            [H - 1, 0],            # Bottom-Left
            [H - 1, W - 1]         # Bottom-Right
        ], dtype=jnp.int32)
        corner_idx = jax.random.randint(k_star, shape=(), minval=0, maxval=4)
        star_yx = corner_candidates[corner_idx]
        grid = self._place(grid, star_yx, TILES_REGISTRY[Tiles.STAR, Colors.GREEN])

        # Agent
        v = params.view_size
        h = v // 2

        gy, gx = goal_yx[0], goal_yx[1]
        y_min, y_max = y0 + 1, y1 - 1
        x_min = jax.lax.select(goal_in_left, x0 + 1, midx + 1)
        x_max = jax.lax.select(goal_in_left, midx - 1, x1 - 1)

        ay = jnp.clip(gy + h, y_min, y_max)
        ax = jnp.clip(gx,       x_min, x_max)

        agent = AgentState(
            position=jnp.stack([ay, ax], axis=0),
            direction=jnp.asarray(0, jnp.int32)  # UP
        )

        carry = SwapCarry(
            star_reached=jnp.asarray(False, dtype=jnp.bool_),
            star_reached_step=jnp.asarray(-1, dtype=jnp.int32),
            swap_done=jnp.asarray(False, dtype=jnp.bool_),
            goal_yx=goal_yx,
            star_yx=star_yx,
            empty_tile=empty_tile,
            p_swap_test=jnp.asarray(params.swap_prob, dtype=jnp.float32),
            doors_yx=jnp.stack([left_door_yx, right_door_yx], axis=0),
            door_close_delay=jnp.asarray(-1, dtype=jnp.int32),
        )

        return State(
            key=key,
            step_num=jnp.asarray(0, dtype=jnp.int32),
            grid=grid,
            agent=agent,
            goal_encoding=_goal_encoding,
            rule_encoding=_rule_encoding,
            carry=carry,
        )

    def handle_star_reach(self, state: State[SwapCarry], params: EnvParams) -> State[SwapCarry]:
        # Unpack params
        testing = params.testing
        random_delay_enabled = params.random_door_close_delay
        max_delay = params.door_close_delay
        
        carry = state.carry
        
        # Check if star is adjacent and facing, and this is the FIRST time we process it
        is_at_star = self._is_adjacent_and_facing(state.agent, carry.star_yx)
        trigger_first_reach = is_at_star & (~carry.star_reached)

        # --- 1. HANDLE GOAL SWAP & DELAY SAMPLING (On First Reach) ---
        def _process_first_reach(_state: State[SwapCarry]) -> State[SwapCarry]:
            # Sample keys
            _key, k_move, k_pick, k_delay = jax.random.split(_state.key, 4)

            # Determine delay for this specific episode
            sampled_delay = jax.random.randint(k_delay, shape=(), minval=1, maxval=max_delay + 1)
            fixed_delay = jnp.asarray(max_delay, dtype=jnp.int32)
            
            actual_delay = jax.lax.select(
                jnp.asarray(random_delay_enabled),
                sampled_delay,
                fixed_delay
            )

            # B. Remove Star
            g1 = _state.grid.at[_state.carry.star_yx[0], _state.carry.star_yx[1]].set(_state.carry.empty_tile)

            # C. Determine Room Masks for Swapping
            H, W = _state.grid.shape[0], _state.grid.shape[1]
            y0, y1, x0, x1, midx = self._get_building_bounds(H, W)
            Y, X = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
            
            left_mask  = (Y > y0) & (Y < y1) & (X > x0) & (X < midx)
            right_mask = (Y > y0) & (Y < y1) & (X > midx) & (X < x1)

            # Identify "Other" Room
            gx = _state.carry.goal_yx[1]
            goal_is_left = gx < midx
            target_mask = jax.lax.select(goal_is_left, right_mask, left_mask)

            # D. Sample New Goal Position (Gumbel-Max)
            flat_mask = target_mask.reshape(-1)
            logits = jnp.where(flat_mask, 0.0, -1e9)
            flat_idx = jax.random.categorical(k_pick, logits)
            ny, nx = jnp.divmod(flat_idx, W)
            new_yx = jnp.stack([ny, nx])

            # E. Decide if Swap Happens
            do_move = jax.lax.select(
                jnp.asarray(testing),
                jax.random.bernoulli(k_move, p=_state.carry.p_swap_test),
                jnp.asarray(False, dtype=jnp.bool_)
            )

            # F. Update Carry (Store the actual delay!)
            def _update_carry(c, new_g_yx):
                return dataclasses.replace(c, 
                    star_reached=True, 
                    star_reached_step=_state.step_num, 
                    swap_done=True, 
                    goal_yx=new_g_yx,
                    door_close_delay=actual_delay
                )

            # G. Update Grid
            def _move(s):
                gy, gx = s.carry.goal_yx[0], s.carry.goal_yx[1]
                g2 = g1.at[gy, gx].set(s.carry.empty_tile)
                g2 = g2.at[ny, nx].set(TILES_REGISTRY[Tiles.GOAL, Colors.GREEN])
                return dataclasses.replace(s, key=_key, grid=g2, carry=_update_carry(s.carry, new_yx))

            def _stay(s):
                gy, gx = s.carry.goal_yx[0], s.carry.goal_yx[1]
                g2 = g1.at[gy, gx].set(TILES_REGISTRY[Tiles.GOAL, Colors.GREEN])
                return dataclasses.replace(s, key=_key, grid=g2, carry=_update_carry(s.carry, s.carry.goal_yx))

            return jax.lax.cond(do_move, _move, _stay, _state)

        # Apply First Reach Logic if triggered
        state = jax.lax.cond(trigger_first_reach, _process_first_reach, lambda s: s, state)

        # --- 2. HANDLE DOOR CLOSING (Based on Stored Delay) ---
        # Logic: If star has been reached, check if current_step matches target time
        # Target Time = star_reached_step + stored_episode_delay
        
        target_step = state.carry.star_reached_step + state.carry.door_close_delay
        should_close_now = state.carry.star_reached & (state.step_num == target_step)

        def _close_doors(s):
            DOOR_CLOSED = TILES_REGISTRY[Tiles.DOOR_CLOSED, Colors.GREEN]
            ldy, ldx = s.carry.doors_yx[0]
            rdy, rdx = s.carry.doors_yx[1]
            g_closed = s.grid.at[ldy, ldx].set(DOOR_CLOSED).at[rdy, rdx].set(DOOR_CLOSED)
            return s.replace(grid=g_closed)

        state = jax.lax.cond(should_close_now, _close_doors, lambda s: s, state)

        return state

    def _get_observer_view(self, grid: jnp.ndarray, view_size: int) -> jnp.ndarray:
        """Generates the view for the fixed observer at (9,5) facing up."""
        observer_state = AgentState(
            position=jnp.array([9, 5], dtype=jnp.int32),
            direction=jnp.array(0, dtype=jnp.int32)  # 0 = UP
        )
        return transparent_field_of_view(grid, observer_state, view_size, view_size)

    def reset(self, params: EnvParams, key: jax.Array) -> TimeStep[SwapCarry]:
        state = self._generate_problem(params, key)
        agent_layer = get_agent_layer(state.agent, state.grid)
        visual_grid = jnp.where(
            agent_layer != TILES_REGISTRY[Tiles.EMPTY, Colors.EMPTY], 
            agent_layer, 
            state.grid
        )

        obs_main = transparent_field_of_view(visual_grid, state.agent, params.view_size, params.view_size)
        obs_observer = self._get_observer_view(visual_grid, params.view_size)
        combined_obs = {}
        combined_obs["p_img"] = obs_main
        combined_obs["o_img"] = obs_observer
        return TimeStep(
            state=state,
            step_type=StepType.FIRST,
            reward=jnp.asarray(0.0),
            discount=jnp.asarray(1.0),
            observation=combined_obs,
            allocentric_obs=visual_grid,
        )

    def step(
        self,
        params: EnvParams,
        timestep: TimeStep[SwapCarry],
        action: IntOrArray,
    ) -> TimeStep[SwapCarry]:
        
        # 1. Physics & Logic (performed on the CLEAN grid)
        new_grid, new_agent, changed_pos = take_action(timestep.state.grid, timestep.state.agent, action)
        new_grid, new_agent = check_rule(timestep.state.rule_encoding, new_grid, new_agent, action, changed_pos)

        new_state = timestep.state.replace(
            grid=new_grid,
            agent=new_agent,
            step_num=timestep.state.step_num + 1,
        )

        # 2. Swap tiles & Handle Doors
        new_state = self.handle_star_reach(new_state, params)

        # 3. Observations
        agent_layer = get_agent_layer(new_state.agent, new_grid)
        visual_grid = jnp.where(
            agent_layer != TILES_REGISTRY[Tiles.EMPTY, Colors.EMPTY], 
            agent_layer, 
            new_state.grid
        )

        obs_main = transparent_field_of_view(visual_grid, new_state.agent, params.view_size, params.view_size)
        obs_observer = self._get_observer_view(visual_grid, params.view_size)
        combined_obs = {}
        combined_obs["p_img"] = obs_main
        combined_obs["o_img"] = obs_observer

        terminated = check_goal(new_state.goal_encoding, new_state.grid, new_state.agent, action, changed_pos)
        
        truncated = jnp.equal(new_state.step_num, params.max_steps)
        reward = jax.lax.select(terminated, 1.0 - 0.9 * (new_state.step_num / params.max_steps), 0.0)
        
        step_type = jax.lax.select(terminated | truncated, StepType.LAST, StepType.MID)
        discount = jax.lax.select(terminated, jnp.asarray(0.0), jnp.asarray(1.0))

        return TimeStep(
            state=new_state,
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=combined_obs,
            allocentric_obs=visual_grid,
        )
        
    def observation_shape(self, params: EnvParamsT) -> tuple[int, int, int] | dict[str, Any]:
        if not(params.use_color):
            return params.view_size, params.view_size
        else:
            return params.view_size, params.view_size, NUM_LAYERS