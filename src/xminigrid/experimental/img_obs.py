# jit-compatible RGB observations. Currently experimental!
# if it proves useful and necessary in the future, I will consider rewriting env.render in such style also
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from ..benchmarks import load_bz2_pickle, save_bz2_pickle
from ..core.constants import NUM_COLORS, NUM_LAYERS, TILES_REGISTRY
from ..rendering.rgb_render import render_tile
from ..wrappers import Wrapper

CACHE_PATH = os.environ.get("XLAND_MINIGRID_CACHE", os.path.expanduser("~/.xland_minigrid"))
FORCE_RELOAD = os.environ.get("XLAND_MINIGRID_RELOAD_CACHE", False)


def build_cache(tiles: np.ndarray, tile_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      cache:        [Ty, Tx, ts, ts, 3]          (no agent)
      agent_cache:  [4, Ty, Tx, ts, ts, 3]       (agent overlaid, 4 directions)
    """
    Ty, Tx = tiles.shape[:2]
    cache = np.zeros((Ty, Tx, tile_size, tile_size, 3), dtype=np.uint8)
    agent_cache = np.zeros((4, Ty, Tx, tile_size, tile_size, 3), dtype=np.uint8)

    for y in range(Ty):
        for x in range(Tx):
            base = render_tile(
                tile=tuple(tiles[y, x]),
                agent_direction=None,
                highlight=False,
                tile_size=int(tile_size),
            )
            cache[y, x] = base

            for d in range(4):
                with_agent = render_tile(
                    tile=tuple(tiles[y, x]),
                    agent_direction=d,       # <-- render 4 orientations
                    highlight=False,
                    tile_size=int(tile_size),
                )
                agent_cache[d, y, x] = with_agent

    return cache, agent_cache

TILE_SIZE = 32
cache_path = os.path.join(CACHE_PATH, "render_cache")

if not os.path.exists(cache_path) or FORCE_RELOAD:
    os.makedirs(CACHE_PATH, exist_ok=True)
    print("Building rendering cache, may take a while...")
    TILE_CACHE, TILE_W_AGENT_CACHE = build_cache(np.asarray(TILES_REGISTRY), tile_size=TILE_SIZE)

    # Flatten (type,color) grid to a single dimension; keep direction as the first axis
    TILE_CACHE = jnp.asarray(TILE_CACHE).reshape(-1, TILE_SIZE, TILE_SIZE, 3)                 # [-1, ts, ts, 3]
    TILE_W_AGENT_CACHE = jnp.asarray(TILE_W_AGENT_CACHE).reshape(4, -1, TILE_SIZE, TILE_SIZE, 3)  # [4, -1, ts, ts, 3]

    print(f"Done. Cache is saved to {cache_path} and will be reused on consequent runs.")
    save_bz2_pickle({"tile_cache": TILE_CACHE, "tile_agent_cache": TILE_W_AGENT_CACHE}, cache_path)

TILE_CACHE = load_bz2_pickle(cache_path)["tile_cache"]
TILE_W_AGENT_CACHE = load_bz2_pickle(cache_path)["tile_agent_cache"]

# rendering with cached tiles
def _render_obs(obs: jax.Array) -> jax.Array:
    view_size = obs.shape[0]

    obs_flat_idxs = obs[:, :, 0] * NUM_COLORS + obs[:, :, 1]
    # render all tiles
    rendered_obs = jnp.take(TILE_CACHE, obs_flat_idxs, axis=0)

    # add agent tile
    agent_tile = TILE_W_AGENT_CACHE[obs_flat_idxs[view_size - 1, view_size // 2]]
    rendered_obs = rendered_obs.at[view_size - 1, view_size // 2].set(agent_tile)
    # [view_size, view_size, tile_size, tile_size, 3] -> [view_size * tile_size, view_size * tile_size, 3]
    rendered_obs = rendered_obs.transpose((0, 2, 1, 3, 4)).reshape(view_size * TILE_SIZE, view_size * TILE_SIZE, 3)

    return rendered_obs


def render_grid_allocentric(
    grid: jax.Array,        # [H, W, 2]  (type, color)
    agent_state: jax.Array, # [3]        (row, col, dir) for THIS timestep
    view_size: int = 7,     # odd is best: agent centered on back edge
    fov_alpha: float = 0.20,
    highlight_rgb: jnp.ndarray | None = None,
) -> jax.Array:
    H, W = grid.shape[:2]
    ts = TILE_SIZE
    vs = int(view_size)
    half = vs // 2  # half-width to either side

    # ------- base tiles -------
    flat_idx = grid[..., 0] * NUM_COLORS + grid[..., 1]      # [H, W]
    rendered = jnp.take(TILE_CACHE, flat_idx, axis=0)        # [H, W, ts, ts, 3] uint8

    # ------- agent pose (0:up, 1:right, 2:down, 3:left) -------
    r = jnp.clip(agent_state[0].astype(jnp.int32), 0, H - 1)
    c = jnp.clip(agent_state[1].astype(jnp.int32), 0, W - 1)
    d = agent_state[2].astype(jnp.int32)

    # agent overlay (supports 1-dir or 4-dir cache)
    agent_tile = (TILE_W_AGENT_CACHE[d % 4, flat_idx[r, c]]
                  if TILE_W_AGENT_CACHE.ndim == 5
                  else TILE_W_AGENT_CACHE[flat_idx[r, c]])
    rendered = rendered.at[r, c].set(agent_tile)

    # ------- FOV mask (agent at back-center; exclude agent tile) -------
    rows = jnp.arange(H)[:, None]      # [H, 1]
    cols = jnp.arange(W)[None, :]      # [1, W]

    # Inclusive ranges (agent included):
    # up:    rows [r-(vs-1), r],        cols [c-half, c+half]
    mask_up    = (rows >= (r - (vs - 1))) & (rows <= r)             & (cols >= (c - half)) & (cols <= (c + half))
    # right: rows [r-half, r+half],      cols [c, c+(vs-1)]
    mask_right = (rows >= (r - half))     & (rows <= (r + half))    & (cols >= c)          & (cols <= (c + (vs - 1)))
    # down:  rows [r, r+(vs-1)],         cols [c-half, c+half]
    mask_down  = (rows >= r)              & (rows <= (r + (vs - 1)))& (cols >= (c - half)) & (cols <= (c + half))
    # left:  rows [r-half, r+half],      cols [c-(vs-1), c]
    mask_left  = (rows >= (r - half))     & (rows <= (r + half))    & (cols >= (c - (vs - 1))) & (cols <= c)

    mask = lax.switch(
        d % 4,
        (lambda: mask_up, lambda: mask_right, lambda: mask_down, lambda: mask_left)
    )  # [H, W] bool


    # ------- blend highlight over FOV tiles only -------
    hi = (jnp.array([255.0, 255.0, 255.0], dtype=jnp.float32)
          if highlight_rgb is None else highlight_rgb.astype(jnp.float32))
    imgf = rendered.astype(jnp.float32)
    m   = mask[:, :, None, None, None]                # [H, W, 1, 1, 1]
    hi  = hi[None, None, None, None, :]               # [1, 1, 1, 1, 3]
    imgf = jnp.where(m, imgf + fov_alpha * (hi - imgf), imgf)
    rendered = imgf.astype(jnp.uint8)

    # ------- stitch tiles -------
    img = rendered.transpose((0, 2, 1, 3, 4)).reshape(H * ts, W * ts, 3)
    return img

class RGBImgObservationWrapper(Wrapper):
    def observation_shape(self, params):
        new_shape = (params.view_size * TILE_SIZE, params.view_size * TILE_SIZE, 3)

        base_shape = self._env.observation_shape(params)
        if isinstance(base_shape, dict):
            assert "img" in base_shape
            obs_shape = {**base_shape, **{"img": new_shape}}
        else:
            obs_shape = new_shape

        return obs_shape

    def __convert_obs(self, timestep):
        if isinstance(timestep.observation, dict):
            assert "img" in timestep.observation
            rendered_obs = {**timestep.observation, **{"img": _render_obs(timestep.observation["img"])}}
        else:
            rendered_obs = _render_obs(timestep.observation)

        timestep = timestep.replace(observation=rendered_obs)
        return timestep

    def reset(self, params, key):
        timestep = self._env.reset(params, key)
        timestep = self.__convert_obs(timestep)
        return timestep

    def step(self, params, timestep, action):
        timestep = self._env.step(params, timestep, action)
        timestep = self.__convert_obs(timestep)
        return timestep
