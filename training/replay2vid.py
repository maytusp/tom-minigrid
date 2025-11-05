
# BEFORE importing jax:
import os
from pathlib import Path
import jax
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np, cv2, xminigrid
from xminigrid.wrappers import DirectionObservationWrapper, GymAutoResetWrapper

def replay_to_video(env_id, traj_dir="trajs", out_dir="videos", fps=16, frame_stride=1):
    env, env_params = xminigrid.make(env_id)
    env = GymAutoResetWrapper(env)
    env = DirectionObservationWrapper(env)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    files = sorted([f for f in os.listdir(traj_dir) if f.endswith(".npz")])
    for i, fn in enumerate(files):
        data = np.load(os.path.join(traj_dir, fn))
        seed, actions = int(data["seed"]), data["actions"]
        rng = jax.random.PRNGKey(seed)  # even on CPU it's fine
        ts = env.reset(env_params, rng)

        frame0 = env.render(env_params, ts)
        H, W = frame0.shape[:2]
        vp = os.path.join(out_dir, f"ep_{i:03d}.mp4")
        writer = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        writer.write(cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR))

        for step, a in enumerate(actions):
            print(f"step {step}")
            ts = env.step(env_params, ts, int(a))
            if step % frame_stride == 0:
                print("start render")
                frame = env.render(env_params, ts)
                print("stop render")
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if bool(ts.last()):
                break
        writer.release()
        print("Saved", vp)

if __name__ == "__main__":
    replay_to_video("MiniGrid-SwapEmpty-13x13", traj_dir="trajs", out_dir="videos", fps=16, frame_stride=1)