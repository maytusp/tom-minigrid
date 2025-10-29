# Sally-Anne-Style Test (with STAR disappearing on reach) — (y, x) consistent

from __future__ import annotations

import dataclasses
import jax
import jax.numpy as jnp
from typing import Tuple

from ...core.constants import TILES_REGISTRY, Colors, Tiles
from ...core.goals import AgentOnTileGoal
from ...core.grid import room, sample_coordinates, sample_direction
from ...core.rules import EmptyRule
from ...environment import Environment, EnvParams
from ...types import AgentState, State

# goals and rules are hardcoded for minigrid envs
_goal_encoding = AgentOnTileGoal(tile=TILES_REGISTRY[Tiles.GOAL, Colors.GREEN]).encode()
_rule_encoding = EmptyRule().encode()[None, ...]


# --- Carry for this environment (stores bookkeeping for Sally-Anne test) ---
@dataclasses.dataclass
class SwapCarry:
    star_reached: jnp.ndarray            # bool scalar: has the agent already "consumed" the STAR?
    swap_done: jnp.ndarray               # bool scalar: has the Sally–Anne swap already been applied?
    goal_yx: jnp.ndarray                 # int32[2] (y, x)
    star_yx: jnp.ndarray                 # int32[2] (y, x)
    square_yx: jnp.ndarray               # int32[4, 2] (y, x) for four squares
    empty_tile: jnp.ndarray              # tile value representing empty floor (captured from room)
    p_swap_test: float = 0.1             # swap probability at test time


class SwapGoalRandom(Environment[EnvParams, SwapCarry]):
    """Four squares, one goal, one star.
    Task: go to STAR first (adjacent & facing) → STAR disappears. Then go to GOAL.
    Test-time Sally–Anne: with prob 0.1 right after STAR is reached, swap the GOAL with a
    randomly chosen SQUARE.
    """

    def default_params(self, **kwargs) -> EnvParams:
        params = EnvParams(height=9, width=9)
        params = params.replace(**kwargs)
        if params.max_steps is None:
            # formula directly taken from MiniGrid
            params = params.replace(max_steps=4 * (params.height * params.width))
        return params

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _is_adjacent_and_facing(agent: AgentState, target_yx: jnp.ndarray) -> jnp.ndarray:
        """True iff agent is adjacent to target and facing it.
        Directions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT. Positions are (y, x)."""
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
        """Place tile at (y, x)."""
        y, x = int(yx[0]), int(yx[1])
        return grid.at[y, x].set(tile)

    # -------------------------
    # Problem generation
    # -------------------------
    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State[SwapCarry]:
        """Spawn 1 goal (GREEN), 1 star (GREEN), and 4 squares (YELLOW, PURPLE, PINK, GREY)
        at random, all in distinct cells. Agent also at a distinct random cell.
        """
        # Build empty room (walls on border)
        grid = room(params.height, params.width)

        # capture an interior 'empty' tile to later remove the star reliably
        empty_tile = grid[1, 1]  # assumes (1,1) is interior floor

        # We need 1 agent + 1 goal + 1 star + 4 squares = 7 distinct free positions
        key, pos_key, dir_key = jax.random.split(key, num=3)
        coords = sample_coordinates(pos_key, grid, num=7)  # int32[7, 2], unique free cells
        # Order: [agent, goal, star, sq1, sq2, sq3, sq4] — each (y, x)
        agent_yx = coords[0]
        goal_yx  = coords[1]
        star_yx  = coords[2]
        squares_yx = coords[3:7]  # [4,2]

        # Place tiles
        grid = self._place(grid, goal_yx,     TILES_REGISTRY[Tiles.GOAL,   Colors.GREEN])
        grid = self._place(grid, star_yx,     TILES_REGISTRY[Tiles.STAR,   Colors.GREEN])
        grid = self._place(grid, squares_yx[0], TILES_REGISTRY[Tiles.SQUARE, Colors.YELLOW])
        grid = self._place(grid, squares_yx[1], TILES_REGISTRY[Tiles.SQUARE, Colors.PURPLE])
        grid = self._place(grid, squares_yx[2], TILES_REGISTRY[Tiles.SQUARE, Colors.PINK])
        grid = self._place(grid, squares_yx[3], TILES_REGISTRY[Tiles.SQUARE, Colors.GREY])

        # Agent
        agent = AgentState(
            position=agent_yx,  # int32[2] (y, x)
            direction=sample_direction(dir_key),
        )

        carry = SwapCarry(
            star_reached=jnp.asarray(False),
            swap_done=jnp.asarray(False),
            goal_yx=goal_yx,
            star_yx=star_yx,
            square_yx=squares_yx,
            empty_tile=empty_tile,
        )

        state = State(
            key=key,
            step_num=jnp.asarray(0),
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
        """If the agent has just reached the STAR (adjacent & facing), remove STAR immediately.
        If testing, with prob p_swap_test, swap GOAL with a random SQUARE. Runs at most once.
        """
        carry = state.carry

        # If we've already processed the star event, do nothing
        def _no_change(_):
            return state

        def _process_once(_):
            # Check current reach condition
            reached_star = self._is_adjacent_and_facing(state.agent, carry.star_yx)

            def _apply(_state: State[SwapCarry]) -> State[SwapCarry]:
                # 1) Remove STAR now
                _key, choose_key, swap_key = jax.random.split(_state.key, 3)
                sy, sx = int(_state.carry.star_yx[0]), int(_state.carry.star_yx[1])
                new_grid = _state.grid.at[sy, sx].set(_state.carry.empty_tile)

                # 2) Maybe swap GOAL with a random SQUARE (testing only)
                idx = jax.random.randint(choose_key, shape=(), minval=0, maxval=4)
                square_yx = _state.carry.square_yx[idx]
                do_swap = jax.random.bernoulli(swap_key, p=_state.carry.p_swap_test) if testing else False

                def _swap(s: State[SwapCarry]) -> State[SwapCarry]:
                    gy, gx = int(s.carry.goal_yx[0]), int(s.carry.goal_yx[1])
                    qy, qx = int(square_yx[0]),       int(square_yx[1])
                    goal_tile   = new_grid[gy, gx]   # read from new_grid (after star removal)
                    square_tile = new_grid[qy, qx]
                    g2 = new_grid.at[gy, gx].set(square_tile)
                    g2 = g2.at[qy, qx].set(goal_tile)
                    new_carry = dataclasses.replace(
                        s.carry,
                        star_reached=jnp.asarray(True),
                        swap_done=jnp.asarray(True),
                        goal_yx=square_yx,  # goal moved to square's former position
                    )
                    return dataclasses.replace(s, key=_key, grid=g2, carry=new_carry)

                def _no_swap(s: State[SwapCarry]) -> State[SwapCarry]:
                    new_carry = dataclasses.replace(
                        s.carry,
                        star_reached=jnp.asarray(True),
                        swap_done=jnp.asarray(False),
                    )
                    return dataclasses.replace(s, key=_key, grid=new_grid, carry=new_carry)

                return jax.lax.cond(do_swap, _swap, _no_swap, _state)

            # If STAR is reached now and not processed before → apply removal (+ maybe swap)
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
