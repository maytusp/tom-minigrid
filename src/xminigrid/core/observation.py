import jax
import jax.numpy as jnp

from ..types import AgentState, GridState
from .constants import Tiles, AgentTiles, TILES_REGISTRY, AGENT_TILES_REGISTRY, Colors
from .grid import align_with_up, check_see_behind

def crop_field_of_view(grid: GridState, agent: AgentState, height: int, width: int) -> jax.Array:
    # TODO: assert height and width are odd and >= 3
    # TODO: in theory we don't need padding from all 4 sides, only for out of bounds sides
    grid = jnp.pad(
        grid,
        pad_width=((height, height), (width, width), (0, 0)),
        constant_values=Tiles.EMPTY,
    )
    # account for padding
    y = agent.position[0] + height
    x = agent.position[1] + width

    # to make slice_size static we compute top left corner of fov rectangle, then slice with static H, W
    start_indices = jax.lax.switch(
        agent.direction,
        (
            lambda: (y - height + 1, x - (width // 2), 0),
            lambda: (y - (height // 2), x, 0),
            lambda: (y, x - (width // 2), 0),
            lambda: (y - (height // 2), x - width + 1, 0),
        ),
    )
    fov_crop = jax.lax.dynamic_slice(grid, start_indices, (height, width, 2))

    return fov_crop


def transparent_field_of_view(grid: GridState, agent: AgentState, height: int, width: int) -> jax.Array:
    fov_grid = crop_field_of_view(grid, agent, height, width) # crop from the entire grid
    fov_grid = align_with_up(fov_grid, agent.direction) # rotate the cropped symbolic image

    # TODO: should we even do this? Agent with good memory can remember what he picked up.
    # WARN: this can overwrite tile the agent is on, GOAL for example.
    # https://github.com/Farama-Foundation/Minigrid/blob/463f07f20ff5cc2843d4b7cdf12c54a73b7d4d1c/minigrid/minigrid_env.py#L618 # noqa
    # Make it so the agent sees what it's carrying
    # We do this by placing the carried object at the agent's position
    # in the agent's partially observable view
    # fov_grid = fov_grid.at[height - 1, width // 2].set(agent.pocket)

    return fov_grid

def generate_viz_mask_minigrid(grid: GridState) -> jax.Array:
    """
    We want invisible area to be the area that is surrounded/closed by tiles
    Generates a visibility mask using a flood fill approach.
    
    High-level logic
    1. Visibility starts at the agent.
    2. Visibility spreads to neighbors (Up/Down/Left/Right) only from tiles that are
       currently visible and transparent.
    3. Opaque tiles become visible when the spread hits them, but they 
       do not propagate the spread further.
    
    """
    H, W = grid.shape[0], grid.shape[1]
    
    # Precompute Transparency Map
    # We need to know which tiles let light pass through.
    # We vmap check_see_behind over the grid coordinates.
    y_coords, x_coords = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij')
    
    def _is_transparent(y, x):
        # check_see_behind returns True if tile is transparent
        return check_see_behind(grid, jnp.array([y, x]))

    # (H, W) boolean array: True where light can pass
    transparency_mask = jax.vmap(jax.vmap(_is_transparent))(y_coords, x_coords)

    # Initialize Visibility Mask
    # Agent is always at (H-1, W//2) in the UP-aligned FOV crop
    agent_pos = (H - 1, W // 2)
    viz_mask = jnp.zeros((H, W), dtype=jnp.bool_)
    viz_mask = viz_mask.at[agent_pos].set(True)

    # Propagation Loop
    # We iterate enough times to cover the grid diameter.
    # 4-way connectivity is used to define closed areas.
    
    def _propagate_step(mask, _):
        # Only transparent tiles that are already visible can propagate light
        active_sources = mask & transparency_mask
        
        # Shift sources in 4 directions to simulate spreading
        up    = jnp.pad(active_sources[1:],  ((0, 1), (0, 0)), constant_values=False)
        down  = jnp.pad(active_sources[:-1], ((1, 0), (0, 0)), constant_values=False)
        left  = jnp.pad(active_sources[:, 1:], ((0, 0), (0, 1)), constant_values=False)
        right = jnp.pad(active_sources[:, :-1], ((0, 0), (1, 0)), constant_values=False)
        
        # Combine shifts
        spread = up | down | left | right
        
        # Update mask: old visible + newly reached tiles
        new_mask = mask | spread
        return new_mask, None

    # Number of iterations: The longest possible Manhattan path is H + W.
    # We run this fixed number of times (compile-time constant).
    max_steps = (H + W)*2
    viz_mask, _ = jax.lax.scan(_propagate_step, viz_mask, None, length=max_steps)

    return viz_mask

# # Direct port from original minigrid:
# # https://github.com/Farama-Foundation/Minigrid/blob/e6f34bee70c5eb45ca9bfa2ea061cf06dd03e7b3/minigrid/core/grid.py#L291C9-L291C20 # noqa
# # but adapted to jax and transposed grid
# # WARN: only works for field of view crop aligned with UP direction, use align_with_up before!
# def generate_viz_mask_minigrid(grid: GridState) -> jax.Array:
#     H, W = grid.shape[0], grid.shape[1]
#     viz_mask = jnp.zeros((H, W), dtype=jnp.bool_)
#     # agent position with UP alignment, always visible
#     viz_mask = viz_mask.at[H - 1, W // 2].set(True)

#     def _forward_set_visible(mask, y, x):
#         mask = mask.at[y, x + 1].set(True)
#         mask = jax.lax.select(
#             y > 0,
#             mask.at[y - 1, x + 1].set(True).at[y - 1, x].set(True),
#             mask,
#         )
#         return mask

#     def _backward_set_visible(mask, y, x):
#         mask = mask.at[y, x - 1].set(True)
#         mask = jax.lax.select(
#             y > 0,
#             mask.at[y - 1, x - 1].set(True).at[y - 1, x].set(True),
#             mask,
#         )
#         return mask

#     # TODO: precompute check_see_behind mask
#     def _forward_body(carry, x):
#         viz_mask, y = carry
#         viz_mask = jax.lax.select(
#             jnp.logical_and(viz_mask[y, x], check_see_behind(grid, jnp.array((y, x)))),
#             _forward_set_visible(viz_mask, y, x),
#             viz_mask,
#         )
#         return (viz_mask, y), None

#     def _backward_body(carry, x):
#         viz_mask, y = carry
#         viz_mask = jax.lax.select(
#             jnp.logical_and(viz_mask[y, x], check_see_behind(grid, jnp.array((y, x)))),
#             _backward_set_visible(viz_mask, y, x),
#             viz_mask,
#         )
#         return (viz_mask, y), None

#     def _main_body(viz_mask, y):
#         (viz_mask, _), _ = jax.lax.scan(f=_forward_body, init=(viz_mask, y), xs=jnp.arange(0, W - 1))
#         (viz_mask, _), _ = jax.lax.scan(f=_backward_body, init=(viz_mask, y), xs=jnp.arange(1, W), reverse=True)
#         return viz_mask, None

#     viz_mask, _ = jax.lax.scan(f=_main_body, init=(viz_mask), xs=jnp.arange(0, H), reverse=True)

#     return viz_mask


# TODO: works well with unroll=16 and random actions, but very slow with PPO even with high unroll!
def minigrid_field_of_view(grid: GridState, agent: AgentState, height: int, width: int) -> jax.Array:
    fov_grid = crop_field_of_view(grid, agent, height, width)
    fov_grid = align_with_up(fov_grid, agent.direction)
    mask = generate_viz_mask_minigrid(fov_grid)
    # set EMPTY as unseen value for all layers (including colors, as EMPTY color has same id value)
    fov_grid = jnp.where(mask[..., None], fov_grid, Tiles.EMPTY)

    # TODO: should we even do this? Agent with good memory can remember what he picked up.
    # WARN: this can overwrite tile the agent is on, GOAL for example.
    # https://github.com/Farama-Foundation/Minigrid/blob/463f07f20ff5cc2843d4b7cdf12c54a73b7d4d1c/minigrid/minigrid_env.py#L618 # noqa
    # Make it so the agent sees what it's carrying
    # We do this by placing the carried object at the agent's position
    # in the agent's partially observable view
    # fov_grid = fov_grid.at[height - 1, width // 2].set(agent.pocket)

    return fov_grid

def get_agent_layer(agent: AgentState, grid: GridState) -> jnp.ndarray:
    """Creates a separate grid containing only the agent tile."""
    # Start with an empty grid
    # We use '0' (Tiles.EMPTY) for empty space
    layer = jnp.zeros_like(grid)
    
    # Maps direction integers to the specific AGENT tile from the registry.
    # MiniGrid Standard: 0=Up, 1=Right, 2=Down, 3=Left
    # We assume Colors.RED is the standard agent color.
    agent_tile = jnp.array([
        AGENT_TILES_REGISTRY[AgentTiles.AGENT_UP, Colors.RED],
        AGENT_TILES_REGISTRY[AgentTiles.AGENT_RIGHT, Colors.RED],
        AGENT_TILES_REGISTRY[AgentTiles.AGENT_DOWN,  Colors.RED],
        AGENT_TILES_REGISTRY[AgentTiles.AGENT_LEFT,  Colors.RED],
    ])[agent.direction]

    # Place the agent on the layer
    layer = layer.at[agent.position[0], agent.position[1]].set(agent_tile)
    
    return layer