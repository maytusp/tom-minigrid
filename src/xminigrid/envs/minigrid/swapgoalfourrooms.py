# Sally-Anne-Style Test (with STAR disappearing on reach) — JAX-safe, (y,x) consistent

from __future__ import annotations

import dataclasses
import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple

from flax import struct  # <-- make carry a JAX pytree

from ...core.constants import TILES_REGISTRY, Colors, Tiles
from ...core.goals import AgentOnTileGoal, check_goal
from ...core.grid import four_rooms, sample_coordinates, sample_direction, cartesian_product_1d
from ...core.rules import EmptyRule, check_rule
from ...core.actions import take_action
from ...core.observation import minigrid_field_of_view as transparent_field_of_view

from ...environment import Environment, EnvParams
from ...types import AgentState, State, TimeStep, StepType, IntOrArray

# goals and rules are hardcoded for minigrid envs
_goal_encoding = AgentOnTileGoal(tile=TILES_REGISTRY[Tiles.GOAL, Colors.GREEN]).encode()
_rule_encoding = EmptyRule().encode()[None, ...]

_allowed_colors = jnp.array(
    (
        Colors.RED,
        Colors.GREEN,
        Colors.BLUE,
        Colors.PURPLE,
        Colors.YELLOW,
        Colors.GREY,
    ),
    dtype=jnp.uint8,
)

# --- Carry for this environment (stores bookkeeping for Sally-Anne test) ---
@struct.dataclass  # <- JAX/Flax-friendly dataclass (pytree)
class SwapCarry:
    star_reached: jnp.ndarray            # bool[] scalar
    swap_done: jnp.ndarray               # bool[] scalar
    goal_yx: jnp.ndarray                 # int32[2] (y, x)
    star_yx: jnp.ndarray                 # int32[2] (y, x)
    empty_tile: jnp.ndarray              # tile dtype (same as grid's entries)
    p_swap_test: jnp.ndarray             # float32[] scalar (e.g., 0.1)


class SwapParams(EnvParams):
    testing: bool = struct.field(pytree_node=False, default=True)
    swap_prob: float = struct.field(pytree_node=False, default=1.0)
    # ---- NEW: door curriculum controls ----
    progress: float = struct.field(pytree_node=False, default=0.0)          # 0 = start of training, 1 = end
    door_open_prob_start: float = struct.field(pytree_node=False, default=0.0)  # p(open) at progress=0
    door_open_prob_end: float = struct.field(pytree_node=False, default=0.0)    # p(open) at progress=1



# number of doors with 4 rooms
_total_doors = 4



class SwapGoalRandom(Environment[EnvParams, SwapCarry]):
    """Four squares, one goal, one star.
    Task: go to STAR first (adjacent & facing) → STAR disappears. Then go to GOAL.
    Test-time Sally–Anne: with prob 0.1 right after STAR is reached, swap the GOAL with a
    randomly chosen SQUARE.
    """
    def num_actions(self, params: EnvParamsT) -> int:
        return 6

    def default_params(self, **kwargs) -> SwapParams:
        params = SwapParams(height=13, width=13)
        params = params.replace(**{k: v for k, v in kwargs.items() if k in {
            "height","width","view_size","max_steps","render_mode",
            "testing","swap_prob",
            "progress","door_open_prob_start","door_open_prob_end"
        }})
        if params.max_steps is None:
            params = params.replace(max_steps=4 * (params.height * params.width))
        return params
    def _sample_doors(self, key: jax.Array, p_open: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns (is_open[4], colors[4]) where each door is open with prob p_open.
        """
        key_color, key_open = jax.random.split(key)
        colors = jax.random.choice(key_color, _allowed_colors, shape=(_total_doors,))
        is_open = jax.random.bernoulli(key_open, p=jnp.asarray(p_open, jnp.float32), shape=(_total_doors,))
        return is_open.astype(jnp.uint8), colors
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
        """
        13x13 four-rooms map; doors with curriculum. Spawn GOAL (BLUE), STAR, AGENT on free tiles.
        No squares. (GOAL may later move to another room after STAR is reached.)
        """
        def _door_tile(color_u8, open_bool):
            color_i = color_u8.astype(jnp.int32)
            closed_row = TILES_REGISTRY[Tiles.DOOR_CLOSED]
            open_row   = TILES_REGISTRY[Tiles.DOOR_OPEN]
            tile_closed = jnp.take(closed_row, color_i, axis=0)
            tile_open   = jnp.take(open_row,   color_i, axis=0)
            return jax.lax.select(open_bool, tile_open, tile_closed)

        # Build four-rooms base
        grid = four_rooms(params.height, params.width)
        roomW, roomH = params.width // 2, params.height // 2

        key, *keys = jax.random.split(key, num=6)
        door_coords = jax.random.randint(keys[0], shape=(_total_doors,), minval=1, maxval=roomW)

        p_open = (1.0 - jnp.asarray(params.progress, jnp.float32)) * jnp.asarray(params.door_open_prob_start, jnp.float32) \
            + (jnp.asarray(params.progress, jnp.float32))        * jnp.asarray(params.door_open_prob_end,   jnp.float32)

        key, k_doors = jax.random.split(key)
        is_open, door_colors = self._sample_doors(k_doors, p_open)

        # Place 4 doors (two vertical splits + two horizontal splits)
        door_idx = 0
        for i in range(0, 2):
            for j in range(0, 2):
                xL = i * roomW
                yT = j * roomH
                xR = xL + roomW
                yB = yT + roomH

                if i + 1 < 2:
                    tile = _door_tile(door_colors[door_idx], is_open[door_idx])
                    grid = grid.at[yT + door_coords[door_idx], xR].set(tile)
                    door_idx += 1

                if j + 1 < 2:
                    tile = _door_tile(door_colors[door_idx], is_open[door_idx])
                    grid = grid.at[yB, xL + door_coords[door_idx]].set(tile)
                    door_idx += 1

        empty_tile = grid[1, 1]  # interior floor exemplar

        # ---- Sample GOAL, STAR, AGENT using free-tile masks ----
        key, k_goal, k_star, k_agent, dir_key = jax.random.split(key, 5)

        # (No special masks for initial placement — free tiles only)
        goal_yx = sample_coordinates(k_goal, grid, num=1)[0]
        grid = self._place(grid, goal_yx, TILES_REGISTRY[Tiles.GOAL, Colors.BLUE])

        star_yx = sample_coordinates(k_star, grid, num=1)[0]
        grid = self._place(grid, star_yx, TILES_REGISTRY[Tiles.STAR, Colors.GREEN])

        agent_yx = sample_coordinates(k_agent, grid, num=1)[0]
        agent = AgentState(position=agent_yx, direction=sample_direction(dir_key))

        # In test mode: pick a pose so GOAL is visible; then hide STAR far from agent & goal
        if isinstance(params, SwapParams) and params.testing:
            agent = self._agent_pose_to_see_goal(goal_yx=goal_yx, H=params.height, W=params.width, view_size=params.view_size)
            min_sep = jnp.asarray(params.view_size, jnp.int32)

            new_star_yx = self._pick_star_far_and_hidden(
                key,
                goal_yx=goal_yx,
                agent=agent,
                squares_yx=jnp.zeros((0, 2), dtype=jnp.int32),  # no squares
                H=params.height,
                W=params.width,
                view_size=params.view_size,
                min_sep=min_sep,
            )

            sy = jnp.asarray(star_yx[0], jnp.int32); sx = jnp.asarray(star_yx[1], jnp.int32)
            grid = grid.at[sy, sx].set(empty_tile)
            nsy = jnp.asarray(new_star_yx[0], jnp.int32); nsx = jnp.asarray(new_star_yx[1], jnp.int32)
            grid = grid.at[nsy, nsx].set(TILES_REGISTRY[Tiles.STAR, Colors.GREEN])
            star_yx = new_star_yx

        carry = SwapCarry(
            star_reached=jnp.asarray(False, dtype=jnp.bool_),
            swap_done=jnp.asarray(False, dtype=jnp.bool_),
            goal_yx=goal_yx,
            star_yx=star_yx,
            empty_tile=empty_tile,
            p_swap_test=jnp.asarray(params.swap_prob, dtype=jnp.float32),
        )

        state = State(
            key=key,
            step_num=jnp.asarray(0, dtype=jnp.int32),
            grid=grid,
            agent=agent,
            goal_encoding=_goal_encoding,
            rule_encoding=_rule_encoding,
            carry=carry,
        )
        return state

    # -------------------------
    # STAR removal + Sally–Anne swap hook (call from your step)
    # -------------------------
    def maybe_swap_after_star(self, state: State[SwapCarry], testing: bool) -> State[SwapCarry]:
        """
        On first STAR reach: remove STAR immediately.
        If testing, with prob p_swap_test, MOVE the GOAL to a random free cell in a room
        different from the current goal room (one of the other three). Then activate GOAL (GREEN).
        """
        carry = state.carry

        def _no_change(_):
            return state

        def _process_once(_):
            reached_star = self._is_adjacent_and_facing(state.agent, carry.star_yx)

            def _apply(_state: State[SwapCarry]) -> State[SwapCarry]:
                _key, k_room, k_pick = jax.random.split(_state.key, 3)

                # 1) Remove STAR now (JAX-safe indices)
                sy = jnp.asarray(_state.carry.star_yx[0], jnp.int32)
                sx = jnp.asarray(_state.carry.star_yx[1], jnp.int32)
                g0 = _state.grid.at[sy, sx].set(_state.carry.empty_tile)

                # 2) Build 4 room interior masks from grid size
                H, W = g0.shape[0], g0.shape[1]
                roomW, roomH = W // 2, H // 2
                Y = jnp.arange(H, dtype=jnp.int32)[:, None]
                X = jnp.arange(W, dtype=jnp.int32)[None, :]

                # Interiors (exclude border/central walls): (1..roomH-1) etc.
                tl = (Y > 0)       & (Y < roomH) & (X > 0)       & (X < roomW)
                tr = (Y > 0)       & (Y < roomH) & (X > roomW)   & (X < W-1)
                bl = (Y > roomH)   & (Y < H-1)   & (X > 0)       & (X < roomW)
                br = (Y > roomH)   & (Y < H-1)   & (X > roomW)   & (X < W-1)

                masks = jnp.stack([tl, tr, bl, br], axis=0)  # [4,H,W]

                # Identify current goal room index
                gy = _state.carry.goal_yx[0]
                gx = _state.carry.goal_yx[1]
                in_tl = (gy > 0) & (gy < roomH) & (gx > 0)     & (gx < roomW)
                in_tr = (gy > 0) & (gy < roomH) & (gx > roomW) & (gx < W-1)
                in_bl = (gy > roomH) & (gy < H-1) & (gx > 0)   & (gx < roomW)
                # default to br if none matched (shouldn't happen)
                goal_room_idx = jnp.where(in_tl, 0, jnp.where(in_tr, 1, jnp.where(in_bl, 2, 3)))

                # choose one of the other 3 room indices uniformly
                other_choices = jnp.array([[1,2,3],[0,2,3],[0,1,3],[0,1,2]], dtype=jnp.int32)  # [4,3]
                pick_col = jax.random.randint(k_room, shape=(), minval=0, maxval=3)
                other_room_idx = other_choices[goal_room_idx, pick_col]
                target_mask = masks[other_room_idx]  # [H,W] boolean

                # 3) Random free cell in the chosen other room (JAX-safe via sample_coordinates)
                new_goal_yx = sample_coordinates(k_pick, g0, num=1, mask=target_mask)[0]

                do_move = jax.lax.select(
                    jnp.asarray(testing),
                    jax.random.bernoulli(k_room, p=_state.carry.p_swap_test),
                    jnp.asarray(False, dtype=jnp.bool_),
                )

                def _move(s: State[SwapCarry]) -> State[SwapCarry]:
                    # clear old goal to floor, set new to GOAL/GREEN
                    ogy = jnp.asarray(s.carry.goal_yx[0], jnp.int32)
                    ogx = jnp.asarray(s.carry.goal_yx[1], jnp.int32)
                    g1 = g0.at[ogy, ogx].set(s.carry.empty_tile)

                    ny = jnp.asarray(new_goal_yx[0], jnp.int32)
                    nx = jnp.asarray(new_goal_yx[1], jnp.int32)
                    g1 = g1.at[ny, nx].set(TILES_REGISTRY[Tiles.GOAL, Colors.GREEN])

                    new_carry = dataclasses.replace(
                        s.carry,
                        star_reached=jnp.asarray(True, dtype=jnp.bool_),
                        swap_done=jnp.asarray(True, dtype=jnp.bool_),
                        goal_yx=new_goal_yx,
                    )
                    return dataclasses.replace(s, key=_key, grid=g1, carry=new_carry)

                def _stay(s: State[SwapCarry]) -> State[SwapCarry]:
                    ogy = jnp.asarray(s.carry.goal_yx[0], jnp.int32)
                    ogx = jnp.asarray(s.carry.goal_yx[1], jnp.int32)
                    g1 = g0.at[ogy, ogx].set(TILES_REGISTRY[Tiles.GOAL, Colors.GREEN])

                    new_carry = dataclasses.replace(
                        s.carry,
                        star_reached=jnp.asarray(True, dtype=jnp.bool_),
                        swap_done=jnp.asarray(False, dtype=jnp.bool_),
                    )
                    return dataclasses.replace(s, key=_key, grid=g1, carry=new_carry)

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

    def step(
        self,
        params: EnvParams,
        timestep: TimeStep[SwapCarry],
        action: IntOrArray,
    ) -> TimeStep[SwapCarry]:
        # 1) Transition (same as base)
        new_grid, new_agent, changed_position = take_action(timestep.state.grid, timestep.state.agent, action)
        new_grid, new_agent = check_rule(timestep.state.rule_encoding, new_grid, new_agent, action, changed_position)

        new_state = timestep.state.replace(
            grid=new_grid,
            agent=new_agent,
            step_num=timestep.state.step_num + 1,
        )

        # 2) STAR removal + optional Sally–Anne swap/goal activation
        #    If you have no testing flag, this defaults to False.
        testing = getattr(params, "testing", False)
        new_state = self.maybe_swap_after_star(new_state, testing=testing)

        # 3) Observation after possible grid changes
        new_observation = transparent_field_of_view(new_state.grid, new_state.agent, params.view_size, params.view_size)

        # 4) Termination/truncation and reward
        #    Because we only activate GOAL/GREEN after the star, the base goal check enforces star-first completion.
        terminated = check_goal(new_state.goal_encoding, new_state.grid, new_state.agent, action, changed_position)

        assert params.max_steps is not None
        truncated = jnp.equal(new_state.step_num, params.max_steps)

        reward = jax.lax.select(terminated, 1.0 - 0.9 * (new_state.step_num / params.max_steps), 0.0)

        step_type = jax.lax.select(terminated | truncated, StepType.LAST, StepType.MID)
        discount = jax.lax.select(terminated, jnp.asarray(0.0), jnp.asarray(1.0))

        return TimeStep(
            state=new_state,
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=new_observation,
        )
