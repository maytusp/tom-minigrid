# Sally-Anne-Style Test (with STAR disappearing on reach) — JAX-safe, (y,x) consistent

from __future__ import annotations

import dataclasses
import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple

from flax import struct  # <-- make carry a JAX pytree

from ...core.constants import TILES_REGISTRY, Colors, Tiles, NUM_LAYERS
from ...core.goals import AgentOnTileGoal, check_goal
from ...core.grid import room, sample_coordinates, sample_direction, horizontal_line, vertical_line, rectangle
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
    star_reached_step: jnp.ndarray       # int32[] scalar (Added: records T_m)
    swap_done: jnp.ndarray               # bool[] scalar
    goal_yx: jnp.ndarray                 # int32[2]
    star_yx: jnp.ndarray                 # int32[2]
    empty_tile: jnp.ndarray              # tile dtype
    p_swap_test: jnp.ndarray             # float32[] scalar
    doors_yx: jnp.ndarray                # int32[2,2] -> [left_door_yx, right_door_yx]
    saved_agent_pos: jnp.ndarray         # int32[2]
    saved_agent_dir: jnp.ndarray         # int32 scalar
    
class ToMEnvParams(EnvParams):
    testing: bool = struct.field(pytree_node=False, default=True)
    swap_prob: float = struct.field(pytree_node=False, default=1.0)
    use_color: bool = struct.field(pytree_node=False, default=True)

class TwoRooms(Environment[EnvParams, SwapCarry]):
    """Four squares, one goal, one star.
    Task: go to STAR first (adjacent & facing) → STAR disappears. Then go to GOAL.
    Test-time Sally–Anne: with prob 0.1 right after STAR is reached, swap the GOAL with a
    randomly chosen SQUARE.
    """
    def num_actions(self, params: EnvParamsT) -> int:
        return 6

    def default_params(self, **kwargs) -> ToMEnvParams:
        params = ToMEnvParams(height=13, width=13)
        params = params.replace(**{k: v for k, v in kwargs.items() if k in {
            "height","width","view_size","max_steps","render_mode","testing","swap_prob"}})
        if params.max_steps is None:
            params = params.replace(max_steps=4 * (params.height * params.width))
        return params


    @staticmethod
    def _sample_row_from_mask(key: jax.Array, rows: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        """Return one row from `rows` (shape [N,2]) whose index satisfies `mask` (shape [N]).
        JAX-safe: uses integer gather instead of boolean indexing."""
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

        h_room = H // 2
        w_room = W // 2
        
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
        
        H, W = params.height+2, params.width+2
        grid = room(H, W)
        empty_tile = grid[1, 1]

        WALL = TILES_REGISTRY[Tiles.WALL, Colors.GREY]
        DOOR = TILES_REGISTRY[Tiles.DOOR_OPEN, Colors.GREEN]  # closed door, as you set

        # Central 7x7 bounds (inclusive) and vertical split
        # y0, y1 = 2, 6
        # x0, x1 = 2, 6
        # midx = 4
        y0, y1, x0, x1, midx = self._get_building_bounds(H, W) # (3, 8, 3, 8, 5)

        # 7x7 ring + vertical separator
        grid = rectangle(grid, x0, y0, h=(y1 - y0 + 1), w=(x1 - x0 + 1), tile=WALL)
        grid = vertical_line(grid, x=midx, y=y0 + 1, length=(y1 - y0 - 1), tile=WALL)

        # Doors: one for each room, on the OUTER ring (no door between rooms)
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

        # Build boolean masks over the entire grid for room interiors and for the outside.
        Y = jnp.arange(H, dtype=jnp.int32)[:, None]
        X = jnp.arange(W, dtype=jnp.int32)[None, :]

        left_room_mask  = (Y > y0) & (Y < y1) & (X > x0) & (X <  midx)   # interior only
        right_room_mask = (Y > y0) & (Y < y1) & (X > midx) & (X <  x1)   # interior only
        outside_mask    = jnp.logical_not(left_room_mask | right_room_mask)

        # Which room gets the initial GOAL?
        goal_in_left = jax.random.bernoulli(k_pick, p=jnp.asarray(0.5, jnp.float32))
        key, k_goal, k_star, k_agent = jax.random.split(key, 4)

        # 1) Place GOAL (BLUE) inside the chosen room
        goal_mask = jax.lax.select(goal_in_left, left_room_mask, right_room_mask)
        goal_yx = sample_coordinates(k_goal, grid, num=1, mask=goal_mask)[0]
        grid = self._place(grid, goal_yx, TILES_REGISTRY[Tiles.GOAL, Colors.BLUE])

        # 2) Place STAR (GREEN) strictly OUTSIDE the two rooms (never on walls because of free mask)
        star_yx = sample_coordinates(k_star, grid, num=1, mask=outside_mask)[0]
        grid = self._place(grid, star_yx, TILES_REGISTRY[Tiles.STAR, Colors.GREEN])

        # 3) Place AGENT in the SAME ROOM as the goal, with a pose that sees the goal
        # (direction=UP; choose y so goal is within FoV; clamp to that room’s interior)
        v = params.view_size
        h = v // 2

        gy, gx = goal_yx[0], goal_yx[1]
        y_min, y_max = y0 + 1, y1 - 1
        x_min = jax.lax.select(goal_in_left, x0 + 1, midx + 1)
        x_max = jax.lax.select(goal_in_left, midx - 1, x1 - 1)

        ay = jnp.clip(gy + h, y_min, y_max)   # keep goal within [ay - v + 1, ay]
        ax = jnp.clip(gx,       x_min, x_max) # center x on goal but stay inside the room

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
            saved_agent_pos=jnp.zeros((2,), dtype=jnp.int32),
            saved_agent_dir=jnp.asarray(0, dtype=jnp.int32),
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

    def handle_star_reach(self, state: State[SwapCarry], testing: bool) -> State[SwapCarry]:
        carry = state.carry

        def _no_change(_): return state

        def _process_swap(_):
            carry = state.carry
            reached_star = self._is_adjacent_and_facing(state.agent, carry.star_yx)
            trigger_tm = reached_star & (~carry.star_reached) # star reached for the first time

            def _apply(_state: State[SwapCarry]) -> State[SwapCarry]:
                _key, k_move, k_pick = jax.random.split(_state.key, 3)

                # Capture current agent state (to restore later)
                saved_pos = _state.agent.position
                saved_dir = _state.agent.direction
                # Create the "Blind" Agent state (At 0,0, facing UP)
                blind_agent = AgentState(
                    position=jnp.array([1, 1], dtype=jnp.int32),
                    direction=jnp.array(0, dtype=jnp.int32)
                )

                # A. Remove Star & Close Doors
                g1 = _state.grid.at[_state.carry.star_yx[0], _state.carry.star_yx[1]].set(_state.carry.empty_tile)

                # B. Compute Room Interiors using MASKS (Dynamic Safe)
                H, W = _state.grid.shape[0], _state.grid.shape[1]
                y0, y1, x0, x1, midx = self._get_building_bounds(H, W)
                
                Y, X = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
                
                # Define interiors based on walls
                left_mask  = (Y > y0) & (Y < y1) & (X > x0) & (X < midx)
                right_mask = (Y > y0) & (Y < y1) & (X > midx) & (X < x1)

                # Identify "Other" Room
                gx = _state.carry.goal_yx[1]
                goal_is_left = gx < midx
                target_mask = jax.lax.select(goal_is_left, right_mask, left_mask)

                # C. Sample New Position (Gumbel-Max Trick)
                # This works for any grid size without array slicing
                flat_mask = target_mask.reshape(-1)
                # logits: 0 for valid spots, -inf for invalid
                logits = jnp.where(flat_mask, 0.0, -1e9)
                
                flat_idx = jax.random.categorical(k_pick, logits)
                ny, nx = jnp.divmod(flat_idx, W)
                new_yx = jnp.stack([ny, nx])

                do_move = jax.lax.select(
                    jnp.asarray(testing),
                    jax.random.bernoulli(k_move, p=_state.carry.p_swap_test),
                    jnp.asarray(False, dtype=jnp.bool_)
                )

                def _update_carry(c, new_g_yx):
                    return dataclasses.replace(c, 
                        star_reached=True, 
                        star_reached_step=_state.step_num, 
                        swap_done=True, 
                        goal_yx=new_g_yx,
                        saved_agent_pos=saved_pos,
                        saved_agent_dir=saved_dir,
                    )

                def _move(s):
                    gy, gx = s.carry.goal_yx[0], s.carry.goal_yx[1]
                    g2 = g1.at[gy, gx].set(s.carry.empty_tile)
                    g2 = g2.at[ny, nx].set(TILES_REGISTRY[Tiles.GOAL, Colors.GREEN])
                    return dataclasses.replace(s, key=_key, grid=g2, agent=blind_agent, carry=_update_carry(s.carry, new_yx))

                def _stay(s):
                    gy, gx = s.carry.goal_yx[0], s.carry.goal_yx[1]
                    g2 = g1.at[gy, gx].set(TILES_REGISTRY[Tiles.GOAL, Colors.GREEN])
                    return dataclasses.replace(s, key=_key, grid=g2, agent=blind_agent, carry=_update_carry(s.carry, s.carry.goal_yx))

                return jax.lax.cond(do_move, _move, _stay, _state)

            return jax.lax.cond(reached_star, _apply, lambda s: s, state)

        def _process_door(s):
            # Close Doors
            DOOR_CLOSED = TILES_REGISTRY[Tiles.DOOR_CLOSED, Colors.GREEN]
            ldy, ldx = s.carry.doors_yx[0]
            rdy, rdx = s.carry.doors_yx[1]
            g1 = s.grid.at[ldy, ldx].set(DOOR_CLOSED).at[rdy, rdx].set(DOOR_CLOSED)
            
            # Restore Agent from Carry
            # We explicitly ignore whatever physics happened at (0,0) in the previous step
            restored_agent = AgentState(
                position=s.carry.saved_agent_pos,
                direction=s.carry.saved_agent_dir
            )
            
            return s.replace(grid=g1, agent=restored_agent)

        swap_done = jnp.logical_or(carry.star_reached, carry.swap_done)
        new_state = jax.lax.cond(swap_done, _no_change, _process_swap, operand=None)

        close_door = (swap_done) & (state.step_num == new_state.carry.star_reached_step + 1)
        new_state = jax.lax.cond(close_door, _process_door, lambda s: s, operand=new_state)
        return new_state

    def reset(self, params: EnvParams, key: jax.Array) -> TimeStep[EnvCarryT]:
        # Generate the Clean State
        state = self._generate_problem(params, key)

        # Create the Visual Grid
        agent_layer = get_agent_layer(state.agent, state.grid)
        
        visual_grid = jnp.where(
            agent_layer != TILES_REGISTRY[Tiles.EMPTY, Colors.EMPTY], 
            agent_layer, 
            state.grid
        )

        # Generate Observation
        # The first observation now correctly sees the agent
        obs = transparent_field_of_view(visual_grid, state.agent, params.view_size, params.view_size)

        if not(params.use_color):
            obs = obs[:, :, 0]

        return TimeStep(
            state=state,
            step_type=StepType.FIRST,
            reward=jnp.asarray(0.0),
            discount=jnp.asarray(1.0),
            observation=obs,
            allocentric_obs=visual_grid,
        )
    def step(
        self,
        params: EnvParams,
        timestep: TimeStep[SwapCarry],
        action: IntOrArray,
    ) -> TimeStep[SwapCarry]:
        
        # 1. Physics & Logic (performed on the CLEAN grid)
        # We do NOT remove or paint the agent here. The grid stays static regarding the agent.
        new_grid, new_agent, changed_pos = take_action(timestep.state.grid, timestep.state.agent, action)
        new_grid, new_agent = check_rule(timestep.state.rule_encoding, new_grid, new_agent, action, changed_pos)

        new_state = timestep.state.replace(
            grid=new_grid,
            agent=new_agent,
            step_num=timestep.state.step_num + 1,
        )

        # 2. Swap Logic (Goal/Star mechanics)
        testing = getattr(params, "testing", False)
        new_state = self.handle_star_reach(new_state, testing=testing)

        # --- 3. COMPOSITION (The "Separate Grid" Logic) ---
        
        # A. Generate the separate agent layer
        agent_layer = get_agent_layer(new_state.agent, new_grid)
        
        # B. Combine for the Camera (Overlay)
        # Logic: If the agent layer has something (is not EMPTY/0), show the Agent.
        #        Otherwise, show the Environment Grid.
        visual_grid = jnp.where(
            agent_layer != TILES_REGISTRY[Tiles.EMPTY, Colors.EMPTY], 
            agent_layer, 
            new_state.grid
        )


        # C. Generate Observation from the Combined Visual Grid
        # The agent 'sees' the combined version, but the logic operates on the clean version.
        new_obs = transparent_field_of_view(visual_grid, new_state.agent, params.view_size, params.view_size)

        # --- 4. Rewards & Termination ---
        terminated = check_goal(new_state.goal_encoding, new_state.grid, new_state.agent, action, changed_pos)
        
        truncated = jnp.equal(new_state.step_num, params.max_steps)
        reward = jax.lax.select(terminated, 1.0 - 0.9 * (new_state.step_num / params.max_steps), 0.0)
        
        step_type = jax.lax.select(terminated | truncated, StepType.LAST, StepType.MID)
        discount = jax.lax.select(terminated, jnp.asarray(0.0), jnp.asarray(1.0))


        if not(params.use_color):
            new_obs = new_obs[:, :, 0]

        return TimeStep(
            state=new_state,
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=new_obs,
            allocentric_obs=visual_grid,
        )
    def observation_shape(self, params: EnvParamsT) -> tuple[int, int, int] | dict[str, Any]:
        if not(params.use_color):
            return params.view_size, params.view_size
        else:
            return params.view_size, params.view_size, NUM_LAYERS