from __future__ import annotations

import dataclasses
import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple, Optional

# -------------------------------------------------------------------------
# ASSUMED IMPORTS
# -------------------------------------------------------------------------
from ...core.constants import TILES_REGISTRY, Colors, Tiles
from ...core.goals import AgentOnTileGoal, check_goal
from ...core.grid import sample_coordinates, sample_door_coordinates, sample_direction
from ...core.rules import EmptyRule, check_rule
from ...core.actions import take_action
from ...core.observation import minigrid_field_of_view as transparent_field_of_view
from ...core.observation import get_agent_layer
from ...environment import Environment, EnvParams
from ...types import AgentState, State, TimeStep, StepType, IntOrArray

# -------------------------------------------------------------------------
# CONSTANTS & TYPES
# -------------------------------------------------------------------------

ID_CORRIDOR = 0
ID_UNDEFINED = -1

_goal_encoding = AgentOnTileGoal(tile=TILES_REGISTRY[Tiles.GOAL, Colors.GREEN]).encode()
_rule_encoding = EmptyRule().encode()[None, ...]

@struct.dataclass
class SwapCarry:
    star_reached: jnp.ndarray            # bool
    swap_done: jnp.ndarray               # bool
    goal_yx: jnp.ndarray                 # int32[2] (y, x)
    star_yx: jnp.ndarray                 # int32[2] (y, x)
    empty_tile: jnp.ndarray              # tile dtype (floor)
    p_swap_test: jnp.ndarray             # float32
    room_labels: jnp.ndarray             # int32[H, W]

@struct.dataclass
class LayoutInfo:
    grid: jnp.ndarray          
    room_labels: jnp.ndarray   
    num_rooms: jnp.ndarray
    wall_mask: jnp.ndarray     # Boolean mask of valid wall locations

class SwapParams(EnvParams):
    testing: bool = struct.field(pytree_node=False, default=True)
    swap_prob: float = struct.field(pytree_node=False, default=1.0)
    min_num_rooms: int = struct.field(pytree_node=False, default=2)
    max_num_rooms: int = struct.field(pytree_node=False, default=4)

# -------------------------------------------------------------------------
# GENERATOR LOGIC
# -------------------------------------------------------------------------

def generate_corridor_layout(key: jax.Array, height: int, width: int, target_num_rooms: int) -> LayoutInfo:
    # 1. Initialize
    floor_tile = TILES_REGISTRY[Tiles.FLOOR, Colors.GREY] 
    grid = jnp.tile(floor_tile[None, None, :], (height, width, 1)) 
    room_labels = jnp.full((height, width), 1, dtype=jnp.int32)

    # 2. Split Logic (Standard BSP)
    def _perform_split(state, _):
        key, labels, current_count = state
        should_split = current_count < target_num_rooms
        
        k_pick, k_axis, k_width, k_pos, k_next = jax.random.split(key, 5)
        target_id = jax.random.randint(k_pick, shape=(), minval=1, maxval=current_count + 1)
        mask = (labels == target_id)
        
        y_indices = jnp.arange(height)[:, None] * mask
        x_indices = jnp.arange(width)[None, :] * mask
        inf = 1000
        y_min = jnp.min(jnp.where(mask, y_indices, inf))
        y_max = jnp.max(jnp.where(mask, y_indices, -inf))
        x_min = jnp.min(jnp.where(mask, x_indices, inf))
        x_max = jnp.max(jnp.where(mask, x_indices, -inf))
        
        room_h = y_max - y_min + 1
        room_w = x_max - x_min + 1
        
        axis = jax.random.randint(k_axis, shape=(), minval=0, maxval=2)
        axis = jax.lax.select(room_h < 6, 1, axis) 
        axis = jax.lax.select(room_w < 6, 0, axis)
        
        c_width = jax.random.randint(k_width, shape=(), minval=1, maxval=3) 
        
        padding = 2
        split_min_y = y_min + padding
        split_max_y = y_max - padding - c_width
        split_min_x = x_min + padding
        split_max_x = x_max - padding - c_width
        
        valid_h = split_max_y > split_min_y
        valid_v = split_max_x > split_min_x
        is_valid = jax.lax.select(axis == 0, valid_h, valid_v)
        
        s_y = jax.random.randint(k_pos, shape=(), minval=split_min_y, maxval=jnp.maximum(split_min_y+1, split_max_y))
        s_x = jax.random.randint(k_pos, shape=(), minval=split_min_x, maxval=jnp.maximum(split_min_x+1, split_max_x))
        
        new_id = current_count + 1
        
        def _split_horz(lbl):
            Y, X = jnp.indices((height, width))
            in_corridor = (Y >= s_y) & (Y < s_y + c_width) & (X >= x_min) & (X <= x_max) & mask
            in_new_room = (Y >= s_y + c_width) & (Y <= y_max) & (X >= x_min) & (X <= x_max) & mask
            lbl = jnp.where(in_corridor, ID_CORRIDOR, lbl)
            lbl = jnp.where(in_new_room, new_id, lbl)
            return lbl

        def _split_vert(lbl):
            Y, X = jnp.indices((height, width))
            in_corridor = (X >= s_x) & (X < s_x + c_width) & (Y >= y_min) & (Y <= y_max) & mask
            in_new_room = (X >= s_x + c_width) & (X <= x_max) & (Y >= y_min) & (Y <= y_max) & mask
            lbl = jnp.where(in_corridor, ID_CORRIDOR, lbl)
            lbl = jnp.where(in_new_room, new_id, lbl)
            return lbl
        
        new_labels = jax.lax.cond(
            should_split & is_valid,
            lambda l: jax.lax.cond(axis == 0, _split_horz, _split_vert, l),
            lambda l: l,
            labels
        )
        new_count = jax.lax.select(should_split & is_valid, current_count + 1, current_count)
        return (k_next, new_labels, new_count), None

    init_state = (key, room_labels, jnp.array(1, dtype=jnp.int32))
    (key, room_labels, final_count), _ = jax.lax.scan(_perform_split, init_state, None, length=3)
    
    # 3. Calculate Wall Mask (Ground Truth)
    p = room_labels
    room_mask = p > 0
    Y, X = jnp.indices((height, width))
    
    # Check neighbors (Explicit Bounds to prevent wrap-around)
    diff_up = (Y > 0) & (p != jnp.roll(p, 1, axis=0))
    diff_dn = (Y < height - 1) & (p != jnp.roll(p, -1, axis=0))
    diff_lf = (X > 0) & (p != jnp.roll(p, 1, axis=1))
    diff_rt = (X < width - 1) & (p != jnp.roll(p, -1, axis=1))
    
    # A wall is a room pixel that touches a non-room pixel
    wall_mask = room_mask & (diff_up | diff_dn | diff_lf | diff_rt)
    
    # Apply Wall Tiles
    wall_tile = TILES_REGISTRY[Tiles.WALL, Colors.GREY]
    grid = jnp.where(wall_mask[..., None], wall_tile, grid)
    
    return LayoutInfo(grid=grid, room_labels=room_labels, num_rooms=final_count, wall_mask=wall_mask)


# -------------------------------------------------------------------------
# ENVIRONMENT CLASS (Fixed Door Placement)
# -------------------------------------------------------------------------

class SallyAnneRooms(Environment[EnvParams, SwapCarry]):
    def num_actions(self, params: EnvParamsT) -> int:
        return 6

    def default_params(self, **kwargs) -> SwapParams:
        params = SwapParams(height=7, width=7)
        params = params.replace(**{k: v for k, v in kwargs.items() if hasattr(params, k)})
        if params.max_steps is None:
            default_max = (params.height * params.width) // 2
            params = params.replace(max_steps=default_max)
        return params
        
    @staticmethod
    def _is_adjacent_and_facing(agent: AgentState, target_yx: jnp.ndarray) -> jnp.ndarray:
        ay, ax = agent.position[0], agent.position[1]
        ty, tx = target_yx[0], target_yx[1]
        dy, dx = ty - ay, tx - ax
        adjacent = (jnp.abs(dy) + jnp.abs(dx)) == 1
        facing = jnp.logical_or(
            jnp.logical_and(agent.direction == 0, jnp.logical_and(dy == -1, dx == 0)),
            jnp.logical_or(
                jnp.logical_and(agent.direction == 1, jnp.logical_and(dy == 0, dx == 1)),
                jnp.logical_or(
                    jnp.logical_and(agent.direction == 2, jnp.logical_and(dy == 1, dx == 0)),
                    jnp.logical_and(agent.direction == 3, jnp.logical_and(dy == 0, dx == -1))
                )
            )
        )
        return jnp.logical_and(adjacent, facing)

    def _generate_problem(self, params: SwapParams, key: jax.Array) -> State[SwapCarry]:
        key, k_rooms, k_gen, k_doors, k_star, k_goal, k_agent, k_dir = jax.random.split(key, 8)
        
        target_rooms = jax.random.randint(k_rooms, shape=(), minval=params.min_num_rooms, maxval=params.max_num_rooms + 1)
        layout = generate_corridor_layout(k_gen, params.height, params.width, target_rooms)
        
        grid = layout.grid
        labels = layout.room_labels
        wall_mask = layout.wall_mask
        
        # --- Place Doors ---
        door_tile = TILES_REGISTRY[Tiles.DOOR_CLOSED, Colors.GREEN]
        
        def _place_door(curr_grid, i):
            room_exists = (i <= layout.num_rooms)
            candidates = wall_mask & (labels == i) & room_exists
            candidates = candidates.at[0, 0].set(False)
            
            k_d = jax.random.fold_in(k_doors, i)
            door_yx = sample_door_coordinates(k_d, curr_grid, num=1, mask=candidates)[0]
            
            has_cand = jnp.any(candidates)
            safe_yx = jax.lax.select(has_cand, door_yx, jnp.array([0, 0]))
            existing_tile = curr_grid[safe_yx[0], safe_yx[1]]
            tile_to_write = jax.lax.select(has_cand, door_tile, existing_tile)
            return curr_grid.at[safe_yx[0], safe_yx[1]].set(tile_to_write)

        grid = jax.lax.fori_loop(1, 5, lambda i, g: _place_door(g, i), grid)

        # --- Place Objects ---
        star_mask = (labels == ID_CORRIDOR)
        star_yx = sample_coordinates(k_star, grid, num=1, mask=star_mask)[0]
        grid = grid.at[star_yx[0], star_yx[1]].set(TILES_REGISTRY[Tiles.STAR, Colors.GREEN])
        
        goal_mask = (labels > 0)
        goal_yx = sample_coordinates(k_goal, grid, num=1, mask=goal_mask)[0]
        grid = grid.at[goal_yx[0], goal_yx[1]].set(TILES_REGISTRY[Tiles.GOAL, Colors.BLUE])
        
        # --- Place Agent (Same Room as Goal) ---
        floor_tile = TILES_REGISTRY[Tiles.FLOOR, Colors.GREY]
        is_floor = (grid == floor_tile).all(axis=-1)
        
        # 1. Identify the Room ID where the Goal was placed
        gy, gx = goal_yx[0], goal_yx[1]
        goal_room_id = labels[gy, gx]
        
        # 2. Create mask for that specific room
        same_room_mask = (labels == goal_room_id)
        
        # 3. Combine with valid floor locations (Empty tiles only)
        target_mask = is_floor & same_room_mask
        
        # 4. Fallback: If the room is somehow full, use global floor mask
        final_mask = jax.lax.select(jnp.any(target_mask), target_mask, is_floor)

        agent_yx = sample_coordinates(k_agent, grid, num=1, mask=final_mask)[0]
        agent = AgentState(position=agent_yx, direction=sample_direction(k_dir))

        carry = SwapCarry(
            star_reached=jnp.asarray(False, dtype=jnp.bool_),
            swap_done=jnp.asarray(False, dtype=jnp.bool_),
            goal_yx=goal_yx,
            star_yx=star_yx,
            empty_tile=floor_tile, 
            p_swap_test=jnp.asarray(params.swap_prob, dtype=jnp.float32),
            room_labels=labels
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
    # ... (maybe_swap_after_star and step remain unchanged) ...
    def maybe_swap_after_star(self, state: State[SwapCarry], testing: bool) -> State[SwapCarry]:
        carry = state.carry
        already_done = jnp.logical_or(carry.star_reached, carry.swap_done)

        def _process_swap(_state: State[SwapCarry]) -> State[SwapCarry]:
            reached = self._is_adjacent_and_facing(_state.agent, carry.star_yx)

            def _perform_logic(s: State[SwapCarry]) -> State[SwapCarry]:
                new_key, k_swap, k_loc = jax.random.split(s.key, 3)
                
                # Remove Star
                sy, sx = s.carry.star_yx[0], s.carry.star_yx[1]
                grid_temp = s.grid.at[sy, sx].set(s.carry.empty_tile)

                # --- Close All Open Doors ---
                # Define tiles
                open_door_tile = TILES_REGISTRY[Tiles.DOOR_OPEN, Colors.GREEN]
                closed_door_tile = TILES_REGISTRY[Tiles.DOOR_CLOSED, Colors.GREEN]

                # Find masks where the grid contains an open door
                is_open_door = (grid_temp == open_door_tile).all(axis=-1)

                # Replace open doors with closed doors (keep other tiles the same)
                grid_closed_doors = jnp.where(is_open_door[..., None], closed_door_tile, grid_temp)

                # Identify Goal Room
                gy, gx = s.carry.goal_yx[0], s.carry.goal_yx[1]
                current_room_id = s.carry.room_labels[gy, gx]

                # Decide Swap
                should_swap = jax.random.bernoulli(k_swap, s.carry.p_swap_test)
                should_swap = jnp.logical_and(should_swap, jnp.asarray(testing))

                def _do_move_goal(g):
                    target_mask = (s.carry.room_labels > 0) & \
                                  (s.carry.room_labels != current_room_id)
                    new_goal_yx = sample_coordinates(k_loc, g, num=1, mask=target_mask)[0]
                    g = g.at[gy, gx].set(s.carry.empty_tile)
                    ny, nx = new_goal_yx[0], new_goal_yx[1]
                    g = g.at[ny, nx].set(TILES_REGISTRY[Tiles.GOAL, Colors.GREEN])
                    return g, new_goal_yx

                def _do_stay_goal(g):
                    g = g.at[gy, gx].set(TILES_REGISTRY[Tiles.GOAL, Colors.GREEN])
                    return g, s.carry.goal_yx

                final_grid, final_goal_yx = jax.lax.cond(
                    should_swap,
                    _do_move_goal,
                    _do_stay_goal,
                    grid_closed_doors
                )

                new_carry = s.carry.replace(
                    star_reached=jnp.array(True),
                    swap_done=jnp.array(True),
                    goal_yx=final_goal_yx
                )
                return s.replace(grid=final_grid, carry=new_carry, key=new_key)

            return jax.lax.cond(reached, _perform_logic, lambda x: x, _state)

        return jax.lax.cond(already_done, lambda x: x, _process_swap, state)

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

        return TimeStep(
            state=state,
            step_type=StepType.FIRST,
            reward=jnp.asarray(0.0),
            discount=jnp.asarray(1.0),
            observation=obs,
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

        return TimeStep(
            state=new_state,
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=new_obs,
        )