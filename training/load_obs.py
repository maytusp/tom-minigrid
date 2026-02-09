import numpy as np
import os


def get_input_output(file_path, ego_file_path):
    try:
        data = np.load(file_path)
        raw_obs = data["o_sym"]
        observations = raw_obs[:, :, :, 0]
        if ego_file_path:
            ego_data = np.load(ego_file_path)
            raw_ego_obs = ego_data["p_sym"]
            ego_actions = ego_data["action"]
            ego_observations = raw_ego_obs[:, :, :, 0]
        else:
            ego_actions = None
            ego_observations = None
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        exit()

    # 2. Find the index where 12 disappears
    star_idx = -1
    star_seen_previously = False

    for i in range(len(observations)):
        current_frame = observations[i]
        has_star = 12 in current_frame
        
        if has_star:
            star_seen_previously = True
        
        if star_seen_previously and not has_star:
            star_idx = i
            # print(f"FOUND: STAR (12) disappeared at Frame {i}")
            break

    first_door_coord = None
    door_open_frame = -1

    # for finding door coordinates
    first_door_mask = (observations[0] == 10)
    door_coords = list(zip(*np.where(first_door_mask)))
    # We start searching FROM the moment the star was found
    for i in range(star_idx, len(observations)):
        curr_frame = observations[i]
        prev_frame = observations[i-1] # Compare with previous to detect CHANGE
        
        # Logic: 
        # 1. Current cell IS 10 (Open Door)
        # 2. Previous cell was NOT 10 (Closed/Locked or something else)
        # 3. This detects the *event* of opening
        
        # Create a boolean mask of where this transition happened
        just_opened_mask = (curr_frame == 10) & (prev_frame != 10)
        
        # Get coordinates where this mask is True
        rows, cols = np.where(just_opened_mask)
        
        if len(rows) > 0:
            # We found a door opening!
            first_door_coord = (rows[0], cols[0])
            door_open_frame = i
            break

    if first_door_coord is None:
        print("The agent never opened a door after finding the star.")

    if door_open_frame < star_idx:
        print("Agent close and open a door beofre reaching star. We do not count this.")
        first_door_coord = None

    seg_obs = observations[:star_idx+1]
    return observations, ego_observations, ego_actions, seg_obs, star_idx, first_door_coord, door_coords

if __name__ == "__main__":
    _, seg_obs, star_idx, first_door_coord = get_input_output()
    dynamic_log = "Below is the observation log:\n"
    for i, obs in enumerate(seg_obs):
        # Format the grid nicely so the model understands the 2D structure
        formatted_grid = "\n".join([str(row) for row in obs])
        dynamic_log += f"\n--- Time Step {i} ---\n{formatted_grid}\n"

    print(dynamic_log)