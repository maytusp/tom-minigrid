import os
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from xminigrid.core.constants import (
    NUM_COLORS, 
    AGENT_TILES_REGISTRY, 
    Tiles, 
    AgentTiles
)
from ..rendering.rgb_render import render_tile
from ..benchmarks import load_bz2_pickle, save_bz2_pickle

# --- 1. CONFIGURATION ---
TILE_SIZE = 32
CACHE_PATH = os.environ.get("XLAND_MINIGRID_CACHE", os.path.expanduser("~/.xland_minigrid"))
SIMPLE_CACHE_FILE = os.path.join(CACHE_PATH, "render_cache_simple")

# Mapping symbolic IDs (13-16) to MiniGrid directions (0-3)
AGENT_ID_MAPPING = {
    AgentTiles.AGENT_UP:    0, # Up
    AgentTiles.AGENT_RIGHT: 1, # Right
    AgentTiles.AGENT_DOWN:  2, # Down
    AgentTiles.AGENT_LEFT:  3, # Left
}

# --- 2. CACHE BUILDER ---
def build_simple_agent_cache(registry, tile_size=32):
    """
    Builds a flat cache where IDs 13-16 are pre-rendered as Agents on Floor.
    registry: must be AGENT_TILES_REGISTRY (covering IDs 0-16)
    """
    # Flatten registry: [NumTiles, NumColors, 2] -> [N, 2]
    flat_registry = registry.reshape(-1, 2)
    num_total = len(flat_registry)
    
    # Simple 4D cache: [TotalCombinations, H, W, 3]
    cache = np.zeros((num_total, tile_size, tile_size, 3), dtype=np.uint8)

    print(f"Building simple agent cache for {num_total} tiles...")

    for i, (t, c) in enumerate(flat_registry):
        t, c = int(t), int(c)
        
        if t in AGENT_ID_MAPPING:
            # --- AGENT TILE (13-16) ---
            # Render Floor background + Agent Overlay based on ID
            img = render_tile(
                tile=(Tiles.FLOOR, c), 
                agent_direction=AGENT_ID_MAPPING[t], 
                highlight=False, 
                tile_size=tile_size
            )
        else:
            # --- STANDARD TILE (0-12) ---
            img = render_tile(
                tile=(t, c),
                agent_direction=None, 
                highlight=False,
                tile_size=tile_size
            )
        cache[i] = img

    return cache

# --- 3. CACHE LOADING (Separate from main cache) ---
if not os.path.exists(SIMPLE_CACHE_FILE) or os.environ.get("XLAND_MINIGRID_RELOAD_CACHE", False):
    os.makedirs(CACHE_PATH, exist_ok=True)
    
    # We use AGENT_TILES_REGISTRY to ensure we cover IDs 0 through 16
    # Note: Convert to numpy if it's a JAX array
    registry_np = np.array(AGENT_TILES_REGISTRY)
    
    raw_cache = build_simple_agent_cache(registry_np, tile_size=TILE_SIZE)
    
    save_bz2_pickle({"simple_cache": raw_cache}, SIMPLE_CACHE_FILE)
    print(f"Simple Agent Cache saved to {SIMPLE_CACHE_FILE}")

# Load into a distinct global variable
_simple_cache_data = load_bz2_pickle(SIMPLE_CACHE_FILE)
SIMPLE_AGENT_CACHE = jnp.asarray(_simple_cache_data["simple_cache"])

# --- 4. THE RENDERING FUNCTION ---
def _render(grid: jax.Array) -> jax.Array:
    """
    Renders a symbolic grid where agent state is embedded in IDs 13-16.
    No separate agent_state argument is required.
    
    Args:
        grid: [H, W, 2] array of (tile_id, color_id)
        
    Returns:
        image: [H * 32, W * 32, 3] RGB image
    """
    H, W = grid.shape[:2]
    
    # 1. Compute Flat Index (TileID * NUM_COLORS + ColorID)
    # Because we built the cache iterating AGENT_TILES_REGISTRY linearly,
    # this math maps perfectly to the cache indices.
    flat_idx = grid[..., 0] * NUM_COLORS + grid[..., 1]  # Shape: [H, W]

    # 2. Direct Lookup
    # SIMPLE_AGENT_CACHE shape: [N, 32, 32, 3]
    # Result shape:             [H, W, 32, 32, 3]
    rendered = jnp.take(SIMPLE_AGENT_CACHE, flat_idx, axis=0)

    # 3. Stitch into a single large image
    # Transpose to [H, 32, W, 32, 3] -> Reshape to [H*32, W*32, 3]
    img = rendered.transpose((0, 2, 1, 3, 4)).reshape(H * TILE_SIZE, W * TILE_SIZE, 3)
    
    return img