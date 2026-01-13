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
    swap_done: jnp.ndarray               # bool[] scalar
    goal_yx: jnp.ndarray                 # int32[2]
    star_yx: jnp.ndarray                 # int32[2]
    empty_tile: jnp.ndarray              # tile dtype
    p_swap_test: jnp.ndarray             # float32[] scalar
    doors_yx: jnp.ndarray                # int32[2,2] -> [left_door_yx, right_door_yx]
    
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

    # -------------------------
    # Problem generation
    # -------------------------
    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State[SwapCarry]:
        H, W = params.height, params.width
        # assert (H, W) == (13, 13), "This layout assumes a 13x13 grid."

        # Base world with perimeter walls
        grid = room(H, W)
        empty_tile = grid[1, 1]

        WALL = TILES_REGISTRY[Tiles.WALL, Colors.GREY]
        DOOR = TILES_REGISTRY[Tiles.DOOR_CLOSED, Colors.GREEN]  # closed door, as you set

        # Central 7x7 bounds (inclusive) and vertical split
        y0, y1 = 2, 6
        x0, x1 = 2, 6
        midx = 4

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

        # -------- JAX-safe, WALL-SAFE sampling via grid-shaped masks --------
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
        # # Test mode: reposition agent to see GOAL; move STAR far/hidden
        # if isinstance(params, ToMEnvParams) and params.testing:
        #     agent = self._agent_pose_to_see_goal(goal_yx=goal_yx, H=H, W=W, view_size=params.view_size)
        #     min_sep = jnp.asarray(params.view_size, jnp.int32)
        #     new_star_yx = self._pick_star_far_and_hidden(
        #         key, goal_yx=goal_yx, agent=agent,
        #         squares_yx=jnp.zeros((0, 2), dtype=jnp.int32),
        #         H=H, W=W, view_size=params.view_size, min_sep=min_sep,
        #     )
        #     grid = grid.at[star_yx[0], star_yx[1]].set(empty_tile)
        #     grid = grid.at[new_star_yx[0], new_star_yx[1]].set(TILES_REGISTRY[Tiles.STAR, Colors.GREEN])
        #     star_yx = new_star_yx

        carry = SwapCarry(
            star_reached=jnp.asarray(False, dtype=jnp.bool_),
            swap_done=jnp.asarray(False, dtype=jnp.bool_),
            goal_yx=goal_yx,
            star_yx=star_yx,
            empty_tile=empty_tile,
            p_swap_test=jnp.asarray(params.swap_prob, dtype=jnp.float32),
            doors_yx=jnp.stack([left_door_yx, right_door_yx], axis=0),  # <— add this
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

    # -------------------------
    # STAR removal + Sally–Anne swap hook (call from your step)
    # -------------------------
    def maybe_swap_after_star(self, state: State[SwapCarry], testing: bool) -> State[SwapCarry]:
        """
        On first star reach: remove STAR immediately.
        If testing, with prob p_swap_test, MOVE the GOAL to a random free cell in the OTHER room.
        Always activate the GOAL as GREEN at its final location.
        """
        carry = state.carry

        def _no_change(_):
            return state

        def _process_once(_):
            reached_star = self._is_adjacent_and_facing(state.agent, carry.star_yx)

            def _apply(_state: State[SwapCarry]) -> State[SwapCarry]:
                _key, k_move, k_pick = jax.random.split(_state.key, 3)
                # 1) Remove STAR
                g0 = _state.grid.at[_state.carry.star_yx[0], _state.carry.star_yx[1]].set(_state.carry.empty_tile)
                
                # Force-close both doors now
                DOOR_CLOSED_OBJECT = TILES_REGISTRY[Tiles.DOOR_CLOSED, Colors.GREEN]
                ldy, ldx = _state.carry.doors_yx[0, 0].astype(jnp.int32), _state.carry.doors_yx[0, 1].astype(jnp.int32)
                rdy, rdx = _state.carry.doors_yx[1, 0].astype(jnp.int32), _state.carry.doors_yx[1, 1].astype(jnp.int32)
                g1 = g0.at[ldy, ldx].set(DOOR_CLOSED_OBJECT)
                g1 = g1.at[rdy, rdx].set(DOOR_CLOSED_OBJECT)

                # 2) Compute room interiors (deterministic)
                y0, y1, x0, x1, midx = 3, 9, 3, 9, 6
                ys_in = jnp.arange(y0 + 1, y1, dtype=jnp.int32)
                xs_left  = jnp.arange(x0 + 1, midx, dtype=jnp.int32)
                xs_right = jnp.arange(midx + 1, x1, dtype=jnp.int32)
                YYL, XXL = jnp.meshgrid(ys_in, xs_left,  indexing="ij")
                YYR, XXR = jnp.meshgrid(ys_in, xs_right, indexing="ij")
                left_interior  = jnp.stack([YYL.reshape(-1), XXL.reshape(-1)], axis=1)
                right_interior = jnp.stack([YYR.reshape(-1), XXR.reshape(-1)], axis=1)

                gy, gx = _state.carry.goal_yx[0], _state.carry.goal_yx[1]
                in_left = jnp.logical_and(jnp.logical_and(gy >= y0 + 1, gy <= y1 - 1),
                                        jnp.logical_and(gx >= x0 + 1, gx <= midx - 1))
                pool_other = jax.lax.select(in_left, right_interior, left_interior)

                # If we move, choose a random target in the OTHER room
                idx_new = jax.random.randint(k_pick, shape=(), minval=0, maxval=pool_other.shape[0])
                new_yx = pool_other[idx_new]

                do_move = jax.lax.select(
                    jnp.asarray(testing),
                    jax.random.bernoulli(k_move, p=_state.carry.p_swap_test),
                    jnp.asarray(False, dtype=jnp.bool_),
                )

                def _move(s: State[SwapCarry]) -> State[SwapCarry]:
                    gy = jnp.asarray(s.carry.goal_yx[0], jnp.int32)
                    gx = jnp.asarray(s.carry.goal_yx[1], jnp.int32)
                    g2 = g1.at[gy, gx].set(s.carry.empty_tile)  # start from g1 (doors closed)
                    ny = jnp.asarray(new_yx[0], jnp.int32)
                    nx = jnp.asarray(new_yx[1], jnp.int32)
                    g2 = g2.at[ny, nx].set(TILES_REGISTRY[Tiles.GOAL, Colors.GREEN])
                    new_carry = dataclasses.replace(
                        s.carry,
                        star_reached=jnp.asarray(True, dtype=jnp.bool_),
                        swap_done=jnp.asarray(True, dtype=jnp.bool_),
                        goal_yx=new_yx,
                    )
                    return dataclasses.replace(s, key=_key, grid=g2, carry=new_carry)

                def _stay(s: State[SwapCarry]) -> State[SwapCarry]:
                    gy = jnp.asarray(s.carry.goal_yx[0], jnp.int32)
                    gx = jnp.asarray(s.carry.goal_yx[1], jnp.int32)
                    g2 = g1.at[gy, gx].set(TILES_REGISTRY[Tiles.GOAL, Colors.GREEN])  # start from g1
                    new_carry = dataclasses.replace(
                        s.carry,
                        star_reached=jnp.asarray(True, dtype=jnp.bool_),
                        swap_done=jnp.asarray(False, dtype=jnp.bool_),
                    )
                    return dataclasses.replace(s, key=_key, grid=g2, carry=new_carry)

                return jax.lax.cond(do_move, _move, _stay, _state)

            return jax.lax.cond(reached_star, _apply, lambda s: s, state)

        already_processed = jnp.logical_or(carry.star_reached, carry.swap_done)
        return jax.lax.cond(already_processed, _no_change, _process_once, operand=None)

    # -------------------------
    # Optional: success checker (star-first then goal)
    # -------------------------
    def success_after_goal(self, state: State[SwapCarry]) -> jnp.ndarray:
        """True if the agent already reached STAR, and is now adjacent & facing GOAL."""
        return jnp.logical_and(
            state.carry.star_reached,
            self._is_adjacent_and_facing(state.agent, state.carry.goal_yx),
        )

    def _fov_bounds(self, agent: AgentState, view_size: int):
        """Return inclusive (ymin, ymax, xmin, xmax) for the agent's FoV per your renderer.
        Directions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT."""
        ay, ax = agent.position[0], agent.position[1]
        v = view_size
        h = v // 2
        def up(_):
            ymin = ay - v + 1; ymax = ay
            xmin = ax - h;      xmax = ax + h
            return ymin, ymax, xmin, xmax
        def right(_):
            ymin = ay - h;      ymax = ay + h
            xmin = ax;          xmax = ax + v - 1
            return ymin, ymax, xmin, xmax
        def down(_):
            ymin = ay;          ymax = ay + v - 1
            xmin = ax - h;      xmax = ax + h
            return ymin, ymax, xmin, xmax
        def left(_):
            ymin = ay - h;      ymax = ay + h
            xmin = ax - v + 1;  xmax = ax
            return ymin, ymax, xmin, xmax
        return jax.lax.switch(agent.direction, (up, right, down, left), operand=None)

    def _in_rect(self, yx: jnp.ndarray, rect):
        ymin, ymax, xmin, xmax = rect
        y, x = yx[0], yx[1]
        return jnp.logical_and(
            jnp.logical_and(y >= ymin, y <= ymax),
            jnp.logical_and(x >= xmin, x <= xmax),
        )

    def _agent_pose_to_see_goal(self, goal_yx: jnp.ndarray, H: int, W: int, view_size: int) -> AgentState:
        """Pick a pose so goal is visible initially. Use dir=UP; center x on goal, y so goal is in view."""
        v = view_size; h = v // 2
        gy, gx = goal_yx[0], goal_yx[1]
        ay = jnp.clip(gy + h, 1, H - 2)  # interior
        ax = jnp.clip(gx,       1, W - 2)
        return AgentState(position=jnp.stack([ay, ax], 0), direction=jnp.asarray(0, jnp.int32))  # UP

    def _pick_star_far_and_hidden(
        self,
        key: jax.Array,
        goal_yx: jnp.ndarray,
        agent: AgentState,
        squares_yx: jnp.ndarray,
        H: int,
        W: int,
        view_size: int,
        min_sep: int,
    ):
        """Choose a star position outside the agent FoV and at least min_sep (Manhattan) from the goal."""
        # All interior coords
        ys = jnp.arange(1, H - 1, dtype=jnp.int32)
        xs = jnp.arange(1, W - 1, dtype=jnp.int32)
        YY, XX = jnp.meshgrid(ys, xs, indexing="ij")   # [H-2, W-2]
        all_yx = jnp.stack([YY.reshape(-1), XX.reshape(-1)], axis=1)  # [N,2]

        # Masks: not at goal, not at agent, not at any square
        neq_goal = jnp.any(all_yx != goal_yx[None, :], axis=1)
        neq_agent = jnp.any(all_yx != agent.position[None, :], axis=1)
        neq_squares = jnp.all(jnp.any(all_yx[:, None, :] != squares_yx[None, :, :], axis=2), axis=1)

        # Outside FoV
        rect = self._fov_bounds(agent, view_size)
        outside_fov = jnp.logical_not(self._in_rect(all_yx.T, rect).T)  # vectorize via transpose trick

        # Far from goal
        gy, gx = goal_yx[0], goal_yx[1]
        manhattan = jnp.abs(all_yx[:, 0] - gy) + jnp.abs(all_yx[:, 1] - gx)
        far = manhattan >= jnp.asarray(min_sep, jnp.int32)

        valid = jnp.logical_and(jnp.logical_and(neq_goal, neq_agent), jnp.logical_and(neq_squares, outside_fov))
        valid = jnp.logical_and(valid, far)

        # Pick the farthest valid cell; if none valid, pick farthest ignoring FoV
        score_valid   = jnp.where(valid,   manhattan, jnp.full_like(manhattan, -10_000))
        score_backup  = manhattan  # used if no valid
        has_valid = jnp.any(valid)

        idx_valid = jnp.argmax(score_valid)
        idx_backup = jnp.argmax(score_backup)
        idx = jax.lax.select(has_valid, idx_valid, idx_backup)

        return all_yx[idx]
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
        new_state = self.maybe_swap_after_star(new_state, testing=testing)

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