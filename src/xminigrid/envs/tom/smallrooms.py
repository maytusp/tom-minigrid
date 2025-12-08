# New env: Two 4x4 rooms randomly placed against the 9x9 center ring.
# The goal (BLUE→GREEN after star) is in one room; the STAR (GREEN) is outside.
# After STAR is reached, with prob swap_prob (testing=True), move the GOAL to a random
# interior cell in the OTHER room (and activate it GREEN).

from __future__ import annotations

import dataclasses
import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple

from ...core.constants import TILES_REGISTRY, Colors, Tiles
from ...core.goals import AgentOnTileGoal, check_goal
from ...core.grid import room, sample_coordinates, rectangle
from ...core.rules import EmptyRule, check_rule
from ...core.actions import take_action
from ...core.observation import minigrid_field_of_view as transparent_field_of_view

from ...environment import Environment, EnvParams
from ...types import AgentState, State, TimeStep, StepType, IntOrArray

_goal_encoding = AgentOnTileGoal(tile=TILES_REGISTRY[Tiles.GOAL, Colors.GREEN]).encode()
_rule_encoding = EmptyRule().encode()[None, ...]


@struct.dataclass
class RandomRoomsCarry:
    star_reached: jnp.ndarray     # bool[] scalar
    swap_done: jnp.ndarray        # bool[] scalar
    goal_yx: jnp.ndarray          # int32[2]
    star_yx: jnp.ndarray          # int32[2]
    empty_tile: jnp.ndarray       # tile dtype
    p_swap_test: jnp.ndarray      # float32[] scalar
    rooms_tl: jnp.ndarray         # int32[2,2]  top-left (y,x) of the two 4x4 rooms (their outer walls)


class SwapParams2(EnvParams):
    testing: bool = struct.field(pytree_node=False, default=True)
    swap_prob: float = struct.field(pytree_node=False, default=1.0)


class TwoRoomsRandom(Environment[SwapParams2, RandomRoomsCarry]):
    """
    Two 4x4 rooms randomly placed along the inner edge of the central 9x9 ring.
    Rooms do not overlap or touch (>=1 tile separation). Goal starts in one room;
    star is outside both. After star is reached, optionally swap goal to the other room.
    """

    # ----- Small utilities -----
    def num_actions(self, params: EnvParams) -> int:
        return 6

    def default_params(self, **kwargs) -> SwapParams2:
        params = SwapParams2(height=13, width=13)
        params = params.replace(**{k: v for k, v in kwargs.items() if k in {
            "height","width","view_size","max_steps","render_mode","testing","swap_prob"}})
        if params.max_steps is None:
            params = params.replace(max_steps=4 * (params.height * params.width))
        return params

    @staticmethod
    def _is_adjacent_and_facing(agent: AgentState, target_yx: jnp.ndarray) -> jnp.ndarray:
        ay, ax = agent.position[0], agent.position[1]
        ty, tx = target_yx[0], target_yx[1]
        dy, dx = ty - ay, tx - ax
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
        y = jnp.asarray(yx[0], jnp.int32)
        x = jnp.asarray(yx[1], jnp.int32)
        return grid.at[y, x].set(tile)

    @staticmethod
    def _sample_row_from_mask(key: jax.Array, rows: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        idxs = jnp.nonzero(mask, size=rows.shape[0], fill_value=0)[0]
        count = jnp.maximum(jnp.sum(mask).astype(jnp.int32), 1)
        ridx = jax.random.randint(key, shape=(), minval=0, maxval=count)
        return rows[idxs[ridx]]

    @staticmethod
    def _rect_interior_mask(H, W, top_left_yx, h, w):
        y0, x0 = top_left_yx[0], top_left_yx[1]
        Y = jnp.arange(H)[:, None]
        X = jnp.arange(W)[None, :]
        return (Y > y0) & (Y < y0 + h - 1) & (X > x0) & (X < x0 + w - 1)

    @staticmethod
    def _rect_full_mask(H, W, top_left_yx, h, w, pad: int = 0):
        y0, x0 = top_left_yx[0] - pad, top_left_yx[1] - pad
        yh, xw = h + 2 * pad, w + 2 * pad
        Y = jnp.arange(H)[:, None]
        X = jnp.arange(W)[None, :]
        return (Y >= y0) & (Y < y0 + yh) & (X >= x0) & (X < x0 + xw)

    # ----- Problem generation -----
    def _generate_problem(self, params: SwapParams2, key: jax.Array) -> State[RandomRoomsCarry]:
        H, W = params.height, params.width
        assert (H, W) == (13, 13), "This layout assumes a 13x13 grid."

        # Base world with perimeter walls
        grid = room(H, W)
        empty_tile = grid[1, 1]
        WALL = TILES_REGISTRY[Tiles.WALL, Colors.GREY]

        # ---- Central 9x9 ring (inclusive bounds) ----
        # This is centered: rows/cols 2..10 (9 cells)
        y0, y1 = 2, 10
        x0, x1 = 2, 10

        # ---- Enumerate all 4x4 room top-left positions that are flush to the 9x9 edge ----
        RH, RW = 4, 4
        # Flush to TOP edge (room's top wall coincides with y0)
        top_xs = jnp.arange(x0, x1 - RW + 2, dtype=jnp.int32)
        top = jnp.stack([jnp.full_like(top_xs, y0), top_xs], axis=1)

        # Flush to BOTTOM edge (room's bottom wall at y1 → top-left y1-3)
        bot_xs = jnp.arange(x0, x1 - RW + 2, dtype=jnp.int32)
        bot = jnp.stack([jnp.full_like(bot_xs, y1 - RH + 1), bot_xs], axis=1)

        # Flush to LEFT edge (room's left wall at x0)
        left_ys = jnp.arange(y0, y1 - RH + 2, dtype=jnp.int32)
        left = jnp.stack([left_ys, jnp.full_like(left_ys, x0)], axis=1)

        # Flush to RIGHT edge (room's right wall at x1 → top-left x1-3)
        right_ys = jnp.arange(y0, y1 - RH + 2, dtype=jnp.int32)
        right = jnp.stack([right_ys, jnp.full_like(right_ys, x1 - RW + 1)], axis=1)

        all_candidates = jnp.concatenate([top, bot, left, right], axis=0)  # [N,2] top-left (y,x)

        key, k1, k2, k_roommask = jax.random.split(key, 4)

        # Pick room A
        idx1 = jax.random.randint(k1, shape=(), minval=0, maxval=all_candidates.shape[0])
        roomA_tl = all_candidates[idx1]

        # Mask out any candidate whose 4x4 rectangle (plus 1-tile padding) intersects room A
        fullA_pad = self._rect_full_mask(H, W, roomA_tl, RH, RW, pad=1)  # >=1 tile separation
        # Build a boolean per-candidate "valid" mask with pure JAX ops:
        # For each candidate, compute whether its padded full mask intersects fullA_pad.
        # Vectorize by constructing each candidate's mask extents analytically.
        # We'll check overlap via interval intersection on y and x.
        cand_y0 = all_candidates[:, 0]
        cand_x0 = all_candidates[:, 1]
        cand_y1 = cand_y0 + RH - 1
        cand_x1 = cand_x0 + RW - 1

        A_y0, A_x0 = roomA_tl[0], roomA_tl[1]
        A_y1, A_x1 = A_y0 + RH - 1, A_x0 + RW - 1

        # Pad by 1 around each candidate and A for "no touching"
        def overlaps(a0, a1, b0, b1):
            return jnp.logical_not((a1 + 1 < b0 - 1) | (b1 + 1 < a0 - 1))

        y_overlap = overlaps(cand_y0, cand_y1, A_y0, A_y1)
        x_overlap = overlaps(cand_x0, cand_x1, A_x0, A_x1)
        touching_or_overlapping = jnp.logical_and(y_overlap, x_overlap)

        valid2 = jnp.logical_not(touching_or_overlapping)
        # Also block picking the exact same candidate again
        valid2 = valid2.at[idx1].set(False)

        # Fallback if everything got masked (shouldn’t happen, but JAX-safe)
        roomB_tl = self._sample_row_from_mask(k2, all_candidates, valid2)

        rooms_tl = jnp.stack([roomA_tl, roomB_tl], axis=0)

        # ---- Draw the two 4x4 rooms (walls) ----
        # ---- constants for the 9x9 middle & room size ----
        RH, RW = 4, 4
        y0, y1 = 2, 10
        x0, x1 = 2, 10

        WALL = TILES_REGISTRY[Tiles.WALL, Colors.GREY]
        DOOR = TILES_REGISTRY[Tiles.DOOR_CLOSED, Colors.PURPLE]

        # ---- 24 candidates: 4x4 rooms flush to the 9x9 boundary ----
        cands_py = []
        for x in range(x0, x1 - RW + 2):            # top edge
            cands_py.append((y0, x))
        for x in range(x0, x1 - RW + 2):            # bottom edge
            cands_py.append((y1 - RH + 1, x))
        for y in range(y0, y1 - RH + 2):            # left edge
            cands_py.append((y, x0))
        for y in range(y0, y1 - RH + 2):            # right edge
            cands_py.append((y, x1 - RW + 1))
        N = len(cands_py)  # 24
        C = jnp.asarray(cands_py, dtype=jnp.int32)   # [24,2] each row is (y,x)

        # ---- sample two distinct, non-touching (>=1 tile gap) rooms ----
        key, k1, k2 = jax.random.split(key, 3)
        idx1 = jax.random.randint(k1, shape=(), minval=0, maxval=N)

        A_y0 = C[idx1, 0]; A_x0 = C[idx1, 1]
        A_y1 = A_y0 + RH - 1
        A_x1 = A_x0 + RW - 1

        cand_y0 = C[:, 0]; cand_x0 = C[:, 1]
        cand_y1 = cand_y0 + RH - 1
        cand_x1 = cand_x0 + RW - 1

        def overlaps(a0, a1, b0, b1):
            # pad by 1 so rooms cannot touch (>= 1 tile separation)
            return jnp.logical_not((a1 + 1 < b0 - 1) | (b1 + 1 < a0 - 1))

        y_overlap = overlaps(cand_y0, cand_y1, A_y0, A_y1)
        x_overlap = overlaps(cand_x0, cand_x1, A_x0, A_x1)
        touching_or_overlapping = jnp.logical_and(y_overlap, x_overlap)

        valid2 = jnp.logical_not(touching_or_overlapping)
        valid2 = valid2.at[idx1].set(False)

        rows = jnp.arange(N, dtype=jnp.int32)
        idx2 = self._sample_row_from_mask(k2, rows, valid2)

        # ---- reuse *existing* rectangle via lax.switch (static args) ----
        def _make_branch(y_const: int, x_const: int):
            def _branch(g):
                return rectangle(g, x=x_const, y=y_const, h=RH, w=RW, tile=WALL)
            return _branch

        branches = tuple(_make_branch(int(y), int(x)) for (y, x) in cands_py)

        grid = jax.lax.switch(idx1, branches, grid)  # draw room A
        grid = jax.lax.switch(idx2, branches, grid)  # draw room B

        # recover top-lefts (dynamic tensors) for later logic
        roomA_tl = C[idx1]  # int32[2]
        roomB_tl = C[idx2]  # int32[2]
        rooms_tl = jnp.stack([roomA_tl, roomB_tl], axis=0)

        # ---- add doors on the wall that faces "outside" (the flush side) ----
        key, k_door_a, k_door_b = jax.random.split(key, 3)

        def place_door_on_outside_wall(g, tl, k):
            is_top    = tl[0] == y0
            is_bottom = tl[0] == (y1 - RH + 1)
            is_left   = tl[1] == x0
            is_right  = tl[1] == (x1 - RW + 1)

            off4 = jnp.arange(4, dtype=jnp.int32)

            top   = jnp.stack([jnp.full((4,), tl[0],              jnp.int32), tl[1] + off4], axis=1)
            bot   = jnp.stack([jnp.full((4,), tl[0] + RH - 1,     jnp.int32), tl[1] + off4], axis=1)
            left  = jnp.stack([tl[0] + off4, jnp.full((4,), tl[1],            jnp.int32)],   axis=1)
            right = jnp.stack([tl[0] + off4, jnp.full((4,), tl[1] + RW - 1,   jnp.int32)],   axis=1)

            # pick which side via switch( idx, branches, operand )
            side_bool = jnp.stack([is_top, is_bottom, is_left, is_right], 0)
            side_idx = jnp.argmax(side_bool.astype(jnp.int32))

            branches = (
                lambda _: top,
                lambda _: bot,
                lambda _: left,
                lambda _: right,
            )
            candidates = jax.lax.switch(side_idx, branches, operand=None)  # [4,2]

            i = jax.random.randint(k, shape=(), minval=0, maxval=4)
            door_yx = candidates[i]
            g = g.at[door_yx[0], door_yx[1]].set(DOOR)
            return g, door_yx


        grid, doorA_yx = place_door_on_outside_wall(grid, roomA_tl, k_door_a)
        grid, doorB_yx = place_door_on_outside_wall(grid, roomB_tl, k_door_b)


        # ---- Build masks: each room interior and global outside mask ----
        Y = jnp.arange(H)[:, None]
        X = jnp.arange(W)[None, :]

        A_int_mask = self._rect_interior_mask(H, W, roomA_tl, RH, RW)
        B_int_mask = self._rect_interior_mask(H, W, roomB_tl, RH, RW)
        rooms_int_mask = A_int_mask | B_int_mask

        # Outside = anywhere not inside either room interior (walls count as outside)
        outside_mask = jnp.logical_not(rooms_int_mask)

        # ---- Place Goal in one room uniformly; Star strictly outside both rooms ----
        key, k_pick_room, k_goal, k_star, k_agent = jax.random.split(key, 5)
        goal_in_A = jax.random.bernoulli(k_pick_room, p=jnp.asarray(0.5, jnp.float32))

        goal_mask = jax.lax.select(goal_in_A, A_int_mask, B_int_mask)
        goal_yx = sample_coordinates(k_goal, grid, num=1, mask=goal_mask)[0]
        grid = self._place(grid, goal_yx, TILES_REGISTRY[Tiles.GOAL, Colors.BLUE])

        star_yx = sample_coordinates(k_star, grid, num=1, mask=outside_mask)[0]
        grid = self._place(grid, star_yx, TILES_REGISTRY[Tiles.STAR, Colors.GREEN])

        # ---- Place agent inside SAME room as goal, facing UP, with FoV covering goal ----
        v = params.view_size
        h = v // 2
        gy, gx = goal_yx[0], goal_yx[1]

        # Compute interior bounds of the chosen room
        tl = jax.lax.select(goal_in_A, roomA_tl, roomB_tl)
        r_y_min = tl[0] + 1
        r_y_max = tl[0] + RH - 2
        r_x_min = tl[1] + 1
        r_x_max = tl[1] + RW - 2

        ay = jnp.clip(gy + h, r_y_min, r_y_max)
        ax = jnp.clip(gx,       r_x_min, r_x_max)

        agent = AgentState(position=jnp.stack([ay, ax], axis=0),
                           direction=jnp.asarray(0, jnp.int32))  # UP

        carry = RandomRoomsCarry(
            star_reached=jnp.asarray(False, dtype=jnp.bool_),
            swap_done=jnp.asarray(False, dtype=jnp.bool_),
            goal_yx=goal_yx,
            star_yx=star_yx,
            empty_tile=empty_tile,
            p_swap_test=jnp.asarray(params.swap_prob, dtype=jnp.float32),
            rooms_tl=rooms_tl,  # we will rebuild interiors from these on swap
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

    # ----- STAR removal + optional Sally–Anne swap (no doors in this variant) -----
    def maybe_swap_after_star(self, state: State[RandomRoomsCarry], testing: bool) -> State[RandomRoomsCarry]:
        carry = state.carry

        def _no_change(_):
            return state

        def _process_once(_):
            reached_star = self._is_adjacent_and_facing(state.agent, carry.star_yx)

            def _apply(_state: State[RandomRoomsCarry]) -> State[RandomRoomsCarry]:
                _key, k_move, k_pick = jax.random.split(_state.key, 3)
                # 1) Remove STAR
                g0 = _state.grid.at[_state.carry.star_yx[0], _state.carry.star_yx[1]].set(_state.carry.empty_tile)

                # 2) Build interiors for the two rooms
                RH, RW = 4, 4
                A_tl = _state.carry.rooms_tl[0]
                B_tl = _state.carry.rooms_tl[1]

                def rect_points(tl):
                    # tl: int32[2] (top-left y,x of a 4x4 room, outer wall included)
                    # Interior cells are tl + (1..2, 1..2)
                    offsets = jnp.asarray([[1,1],[1,2],[2,1],[2,2]], dtype=jnp.int32)  # shape [4,2]
                    tl = jnp.asarray(tl, dtype=jnp.int32)[None, :]                     # shape [1,2]
                    return tl + offsets                                                # shape [4,2]

                A_int = rect_points(A_tl)  # [4,2] but inner 2x2 -> 4 cells
                B_int = rect_points(B_tl)

                gy, gx = _state.carry.goal_yx[0], _state.carry.goal_yx[1]
                in_A = (gy >= A_tl[0] + 1) & (gy <= A_tl[0] + RH - 2) & \
                       (gx >= A_tl[1] + 1) & (gx <= A_tl[1] + RW - 2)
                pool_other = jax.lax.select(in_A, B_int, A_int)

                # Pick new goal location (uniform over other room)
                idx_new = jax.random.randint(k_pick, shape=(), minval=0, maxval=pool_other.shape[0])
                new_yx = pool_other[idx_new]

                do_move = jax.lax.select(
                    jnp.asarray(testing),
                    jax.random.bernoulli(k_move, p=_state.carry.p_swap_test),
                    jnp.asarray(False, dtype=jnp.bool_),
                )

                def _move(s: State[RandomRoomsCarry]) -> State[RandomRoomsCarry]:
                    gy = jnp.asarray(s.carry.goal_yx[0], jnp.int32)
                    gx = jnp.asarray(s.carry.goal_yx[1], jnp.int32)
                    g1 = g0.at[gy, gx].set(s.carry.empty_tile)
                    ny = jnp.asarray(new_yx[0], jnp.int32)
                    nx = jnp.asarray(new_yx[1], jnp.int32)
                    g1 = g1.at[ny, nx].set(TILES_REGISTRY[Tiles.GOAL, Colors.GREEN])
                    new_carry = dataclasses.replace(
                        s.carry,
                        star_reached=jnp.asarray(True, dtype=jnp.bool_),
                        swap_done=jnp.asarray(True, dtype=jnp.bool_),
                        goal_yx=new_yx,
                    )
                    return dataclasses.replace(s, key=_key, grid=g1, carry=new_carry)

                def _stay(s: State[RandomRoomsCarry]) -> State[RandomRoomsCarry]:
                    gy = jnp.asarray(s.carry.goal_yx[0], jnp.int32)
                    gx = jnp.asarray(s.carry.goal_yx[1], jnp.int32)
                    g1 = g0.at[gy, gx].set(TILES_REGISTRY[Tiles.GOAL, Colors.GREEN])
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

    # ----- Success, FoV helpers reused -----
    def success_after_goal(self, state: State[RandomRoomsCarry]) -> jnp.ndarray:
        return jnp.logical_and(
            state.carry.star_reached,
            self._is_adjacent_and_facing(state.agent, state.carry.goal_yx),
        )

    def _fov_bounds(self, agent: AgentState, view_size: int):
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
        v = view_size; h = v // 2
        gy, gx = goal_yx[0], goal_yx[1]
        ay = jnp.clip(gy + h, 1, H - 2)
        ax = jnp.clip(gx,       1, W - 2)
        return AgentState(position=jnp.stack([ay, ax], 0), direction=jnp.asarray(0, jnp.int32))

    # ----- Step -----
    def step(
        self,
        params: SwapParams2,
        timestep: TimeStep[RandomRoomsCarry],
        action: IntOrArray,
    ) -> TimeStep[RandomRoomsCarry]:
        new_grid, new_agent, changed_position = take_action(timestep.state.grid, timestep.state.agent, action)
        new_grid, new_agent = check_rule(timestep.state.rule_encoding, new_grid, new_agent, action, changed_position)

        new_state = timestep.state.replace(
            grid=new_grid,
            agent=new_agent,
            step_num=timestep.state.step_num + 1,
        )

        testing = getattr(params, "testing", False)
        new_state = self.maybe_swap_after_star(new_state, testing=testing)

        new_observation = transparent_field_of_view(new_state.grid, new_state.agent, params.view_size, params.view_size)

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
