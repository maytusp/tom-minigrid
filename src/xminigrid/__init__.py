from .benchmarks import load_benchmark, registered_benchmarks
from .registration import make, register, registered_environments

# TODO: add __all__
__version__ = "0.9.1"

register(
    id="MiniGrid-ToM-TwoRoomsNoSwap-9x9vs9",
    entry_point="xminigrid.envs.tom.tworooms:TwoRooms",
    height=9,
    width=9,
    view_size=9,
    door_close_delay=1,
    testing=False,
)
register(
    id="MiniGrid-ToM-TwoRoomsSwap-9x9vs9",
    entry_point="xminigrid.envs.tom.tworooms:TwoRooms",
    height=9,
    width=9,
    view_size=9,
    door_close_delay=1,
    testing=True,
)

register(
    id="MiniGrid-ToM-TwoRoomsSwap-9x9vs9-d2",
    entry_point="xminigrid.envs.tom.tworooms:TwoRooms",
    height=9,
    width=9,
    view_size=9,
    door_close_delay=2,
    testing=True,
)

register(
    id="MiniGrid-ToM-TwoRoomsSwap-9x9vs9-d20",
    entry_point="xminigrid.envs.tom.tworooms:TwoRooms",
    height=9,
    width=9,
    view_size=9,
    door_close_delay=20,
    testing=True,
)

register(
    id="MiniGrid-Protagonist-ProcGen-9x9vs9",
    entry_point="xminigrid.envs.tom.protagonist_procgen:SallyAnneRooms",
    height=9,
    width=9,
    view_size=9,
    testing=False,
    use_color=True,
)

register(
    id="MiniGrid-Protagonist-ProcGen-9x9vs9-swap",
    entry_point="xminigrid.envs.tom.protagonist_procgen:SallyAnneRooms",
    height=9,
    width=9,
    view_size=9,
    testing=True,
    use_color=True,
)
