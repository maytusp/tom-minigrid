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
from ...core.grid import room, sample_coordinates, sample_direction
from ...core.rules import EmptyRule, check_rule
from ...core.actions import take_action
from ...core.observation import transparent_field_of_view

from ...environment import Environment, EnvParams
from ...types import AgentState, State, TimeStep, StepType, IntOrArray

# goals and rules are hardcoded for minigrid envs
_goal_encoding = AgentOnTileGoal(tile=TILES_REGISTRY[Tiles.GOAL, Colors.GREEN]).encode()
_rule_encoding = EmptyRule().encode()[None, ...]


# --- Carry for this environment (stores bookkeeping for Sally-Anne test) ---
@struct.dataclass  # <- JAX/Flax-friendly dataclass (pytree)
class SwapCarry:
    star_reached: jnp.ndarray            # bool[] scalar
    swap_done: jnp.ndarray               # bool[] scalar
    goal_yx: jnp.ndarray                 # int32[2] (y, x)
    star_yx: jnp.ndarray                 # int32[2] (y, x)
    square_yx: jnp.ndarray               # int32[4, 2] (y, x) for four squares
    empty_tile: jnp.ndarray              # tile dtype (same as grid's entries)
    p_swap_test: jnp.ndarray             # float32[] scalar (e.g., 0.1)


class SwapParams(EnvParams):
    testing: bool = struct.field(pytree_node=False, default=True)
    swap_prob: float = struct.field(pytree_node=False, default=0.5)




class SwapGoalRandom(Environment[EnvParams, SwapCarry]):
    """Four squares, one goal, one star.
    Task: go to STAR first (adjacent & facing) → STAR disappears. Then go to GOAL.
    Test-time Sally–Anne: with prob 0.1 right after STAR is reached, swap the GOAL with a
    randomly chosen SQUARE.
    """

    def default_params(self, **kwargs) -> SwapParams:
        params = SwapParams(height=9, width=9)
        params = params.replace(**{k: v for k, v in kwargs.items() if k in {
            "height","width","view_size","max_steps","render_mode","testing","swap_prob"}})
        if params.max_steps is None:
            params = params.replace(max_steps=4 * (params.height * params.width))
        return params

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
        """Spawn 1 goal (GREEN), 1 star (GREEN), and 4 squares (YELLOW, PURPLE, PINK, GREY)
        at random, all in distinct cells. Agent also at a distinct random cell.
        """
        # Build empty room (walls on border)
        grid = room(params.height, params.width)

        # capture an interior 'empty' tile to later remove the star reliably
        empty_tile = grid[1, 1]  # assumes (1,1) is interior floor (correct tile dtype)

        # We need 1 agent + 1 goal + 1 star + 4 squares = 7 distinct free positions
        key, pos_key, dir_key = jax.random.split(key, num=3)
        coords = sample_coordinates(pos_key, grid, num=7)  # int32[7, 2], unique free cells
        # Order: [agent, goal, star, sq1, sq2, sq3, sq4] — each (y, x)
        agent_yx = coords[0]
        goal_yx  = coords[1]
        star_yx  = coords[2]
        squares_yx = coords[3:7]  # [4,2]

        # Place tiles
        grid = self._place(grid, goal_yx,       TILES_REGISTRY[Tiles.GOAL,   Colors.BLUE])
        grid = self._place(grid, star_yx,       TILES_REGISTRY[Tiles.STAR,   Colors.GREEN])
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
            star_reached=jnp.asarray(False, dtype=jnp.bool_),
            swap_done=jnp.asarray(False, dtype=jnp.bool_),
            goal_yx=goal_yx,
            star_yx=star_yx,
            square_yx=squares_yx,
            empty_tile=empty_tile,
            p_swap_test=jnp.asarray(params.swap_prob, dtype=jnp.float32),  # <- from params
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
        """If the agent has just reached the STAR (adjacent & facing), remove STAR immediately.
        If testing, with prob p_swap_test, swap GOAL with a random SQUARE. Runs at most once.
        """
        carry = state.carry

        def _no_change(_):
            return state

        def _process_once(_):
            # Check current reach condition
            reached_star = self._is_adjacent_and_facing(state.agent, carry.star_yx)

            def _apply(_state: State[SwapCarry]) -> State[SwapCarry]:
                # 1) Remove STAR now (JAX scalars)
                _key, choose_key, swap_key = jax.random.split(_state.key, 3)
                sy = jnp.asarray(_state.carry.star_yx[0], jnp.int32)
                sx = jnp.asarray(_state.carry.star_yx[1], jnp.int32)
                new_grid = _state.grid.at[sy, sx].set(_state.carry.empty_tile)

                # 2) Maybe swap GOAL with a random SQUARE (testing only)
                idx = jax.random.randint(choose_key, shape=(), minval=0, maxval=4)
                square_yx = _state.carry.square_yx[idx]
                do_swap = jax.lax.select(
                    jnp.asarray(testing),  # convert Python bool to JAX bool
                    jax.random.bernoulli(swap_key, p=_state.carry.p_swap_test),
                    jnp.asarray(False, dtype=jnp.bool_),
                )

                def _activate_goal_grid(g: jnp.ndarray, gy: jnp.ndarray, gx: jnp.ndarray) -> jnp.ndarray:
                    """Set tile at (gy, gx) to GOAL GREEN (JAX-safe indices)."""
                    return g.at[gy, gx].set(TILES_REGISTRY[Tiles.GOAL, Colors.GREEN])


                def _swap(s: State[SwapCarry]) -> State[SwapCarry]:
                    gy = jnp.asarray(s.carry.goal_yx[0], jnp.int32)
                    gx = jnp.asarray(s.carry.goal_yx[1], jnp.int32)
                    qy = jnp.asarray(square_yx[0], jnp.int32)
                    qx = jnp.asarray(square_yx[1], jnp.int32)

                    goal_tile   = new_grid[gy, gx]
                    square_tile = new_grid[qy, qx]
                    g2 = new_grid.at[gy, gx].set(square_tile)
                    g2 = g2.at[qy, qx].set(goal_tile)

                    # move goal to square_yx; move that square to old goal location
                    new_square_yx = s.carry.square_yx.at[idx].set(s.carry.goal_yx)

                    new_carry = dataclasses.replace(
                        s.carry,
                        star_reached=jnp.asarray(True, dtype=jnp.bool_),
                        swap_done=jnp.asarray(True, dtype=jnp.bool_),
                        goal_yx=square_yx,
                        square_yx=new_square_yx,
                    )

                    # Activate the goal at its new location
                    agy = jnp.asarray(new_carry.goal_yx[0], jnp.int32)
                    agx = jnp.asarray(new_carry.goal_yx[1], jnp.int32)
                    g2 = _activate_goal_grid(g2, agy, agx)

                    return dataclasses.replace(s, key=_key, grid=g2, carry=new_carry)


                def _no_swap(s: State[SwapCarry]) -> State[SwapCarry]:
                    new_carry = dataclasses.replace(
                        s.carry,
                        star_reached=jnp.asarray(True, dtype=jnp.bool_),
                        swap_done=jnp.asarray(False, dtype=jnp.bool_),
                    )

                    # **Activate** the goal at its existing location
                    agy = jnp.asarray(new_carry.goal_yx[0], jnp.int32)
                    agx = jnp.asarray(new_carry.goal_yx[1], jnp.int32)
                    g2 = _activate_goal_grid(new_grid, agy, agx)

                    return dataclasses.replace(s, key=_key, grid=g2, carry=new_carry)

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
