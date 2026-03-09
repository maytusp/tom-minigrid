from .benchmarks import load_benchmark, registered_benchmarks
from .registration import make, register, registered_environments

# TODO: add __all__
__version__ = "0.9.1"
# training environment for O and P to learn to solve the task without item swapping, some randomness in the door close delay to prevent overfitting.
register(
    id="MiniGrid-ToM-TwoRoomsNoSwap-9x9vs9",
    entry_point="xminigrid.envs.tom.tworooms:TwoRooms",
    height=9,
    width=9,
    view_size=9,
    door_close_delay=2,
    random_door_close_delay=True,
    apply_swap=False,
)

# eval environments: items swap after 1, 2, 4, or 8 steps (with no randomness)
register(
    id="MiniGrid-ToM-TwoRoomsSwap-9x9vs9",
    entry_point="xminigrid.envs.tom.tworooms:TwoRooms",
    height=9,
    width=9,
    view_size=9,
    door_close_delay=1,
    random_door_close_delay=False,
    apply_swap=True,
)

register(
    id="MiniGrid-ToM-TwoRoomsSwap-9x9vs9-d2",
    entry_point="xminigrid.envs.tom.tworooms:TwoRooms",
    height=9,
    width=9,
    view_size=9,
    door_close_delay=2,
    random_door_close_delay=False,
    apply_swap=True,
)


# ProcGen Environments (currently not used)
register(
    id="MiniGrid-Protagonist-ProcGen-9x9vs9",
    entry_point="xminigrid.envs.tom.protagonist_procgen:SallyAnneRooms",
    height=9,
    width=9,
    view_size=9,
    apply_swap=False,
    use_color=True,
)

register(
    id="MiniGrid-Protagonist-ProcGen-9x9vs9-swap",
    entry_point="xminigrid.envs.tom.protagonist_procgen:SallyAnneRooms",
    height=9,
    width=9,
    view_size=9,
    apply_swap=True,
    use_color=True,
)
