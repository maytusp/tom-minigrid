import os
import glob
from openai import OpenAI  # Changed from google.generativeai
from .load_obs import get_input_output

# 1. Load the API key
try:
    with open("/home/maytus/tom/test_llm/cgptk.txt", "r") as f:
        api_key = f.read().strip() 
except FileNotFoundError:
    print("Error: 'mykey.txt' not found. Please create the file with your API key.")
    exit()

# 2. Initialize the OpenAI Client
client = OpenAI(api_key=api_key)
model_name = model="gpt-5.2"


static_instruction = """
### ROLE & CONTEXT
You are a fixed observer agent, watching a protagonist agent (P) in a symbolic 9x9 grid world. 
P is trained via RL to reach the STAR (12) first and then reach the GOAL (6). 
After STAR is reached, STAR will disappear and GOAL will become reachable.
GOAL cannot be achieved if STAR is not reached. 

### GRID DEFINITIONS
The grid is 0-indexed:
- Top-left is (0,0).
- Bottom-right is (8,8).
- Format is (row, column).

### LEGEND (Value_dict)
0: INVISIBLE, 1: FLOOR, 2: WALL, 3: BALL, 4: SQUARE, 
5: PYRAMID, 6: GOAL, 7: KEY, 8: DOOR_LOCKED, 
9: DOOR_CLOSED, 10: DOOR_OPEN, 11: HEX, 12: STAR

### AGENT IDENTIFIERS
P can see 9x9 grid in front of itself. The middle column in the bottom row of its visual field is the observation of itself.
P is defined by the following values depending on facing orientation.
- 13 (AGENT_UP)
- 14 (AGENT_RIGHT)
- 15 (AGENT_DOWN)
- 16 (AGENT_LEFT)

### OBSERVABILITY
Agents (both observer or protagonist) can observe area inside their 9x9 visual field unless the area is completely closed by wall or a closed door.
Examples:
1. You and P can observe the entire room area surrounded by and open door and wall if it faces toward that room.
2. You and P cannot observe the entire room if it is surrounded by a closed door and wall.

### YOUR TASK
Given a part of an episode from start until P reaching STAR, you have to predict what door will P open first.
You have to return the coordinate of the cell containing the opened door (value = 10).

### INPUT FORMAT
A part of an episode represented by a sequence of symbolic observation (a symbolic video).
This part ends when P raches the STAR.

### OUTPUT FORMAT
Provide only the (row, column) coordinate tuple in brackets. Do not write explanations.
Example: (2, 3)
"""

file_dir =  "./logs/eval_trajs/tworoom_swap/"
ego_file_path = None # "./logs/eval_trajs/tworoom_noswap/ep_029_observer_sym.npz"
file_list_all = sorted(glob.glob(os.path.join(file_dir, '*.npz')))
file_list = []
num_eval_episodes = 20
doorloc_acc = 0.0
dooropen_acc = 0.0

for file_path in file_list_all:
    if len(file_list) >= num_eval_episodes:
        break
    total_observations, ego_observations, ego_actions, seg_obs, star_idx, first_door_coord, door_coords = get_input_output(file_path, ego_file_path)
    
    if first_door_coord is not None:
        data = [total_observations, ego_observations, ego_actions, seg_obs, star_idx, first_door_coord, door_coords]
        file_list.append((file_path, data))

print(f"Successfully collected {len(file_list)} episodes.")

for (file_path, data) in file_list:
    [total_observations, ego_observations, ego_actions, seg_obs, star_idx, first_door_coord, door_coords] = data
    observations_data = seg_obs
    if ego_file_path:
        action_map = {
                0: "go forward",
                1: "turn right",
                2: "turn left",
                3: "unused", 
                4: "unused",
                5: "open/close door"
            }

        active_actions = [f"{k}: {v}" for k, v in action_map.items() if "unused" not in v]
        action_legend = ", ".join(active_actions)
        ego_log = f"""
        You have access to first-person experience that you played with before in <start_fp> <stop_fp>.
        The trajectory inside <start_fp> <stop_fp> is your first-person memory, not trajectory of Agent P.
        The frame of reference in <start_fp> <stop_fp> is an egocentric frame. You always observe what is in front of you.
        Action Definition Mapping:
        {action_legend}
        (Note: Actions 3 and 4 are not currently used)


        The format of experience is:
        --- timestep t ---
        frame t: 9x9 array
        action t: <action_integer>
        \n
        <start_fp>
        """
        for ego_t, (ego_obs, ego_act) in enumerate(zip(ego_observations, ego_actions)):
            formatted_ego_grid = "\n".join([str(row) for row in ego_obs])
            ego_log += f"--- timestep {ego_t} ---\n"
            ego_log += f"frame {ego_t}:\n{formatted_ego_grid}\n"
            ego_log += f"action {ego_t}: {ego_act}\n\n"
            
        ego_log += "<stop_fp>\n"
    else:
        ego_log = ""

        print("ego_log", ego_log)
    dynamic_log = "Below is the observation log of the Agent P inside <start_observe> <stop_observe>:\n <start_observe> \n"

    for i, obs in enumerate(observations_data):
        # Format the grid nicely so the model understands the 2D structure
        formatted_grid = "\n".join([str(row) for row in obs])
        dynamic_log += f"\n--- Time Step {i} ---\n{formatted_grid}\n"
    last_frame = observations_data[-1]
    dynamic_log += "<stop_observe>"
    # This is for Gemini: final_prompt = static_instruction + dynamic_log
    final_log = ego_log + dynamic_log
    messages_to_send = [
        {
            "role": "system", 
            "content": static_instruction  # Your role and context rules
        },
        {
            "role": "user", 
            "content": final_log         # Your actual grid data
        }
    ]
    # Send it to OpenAI
    response = client.chat.completions.create(
        model=model_name,
        messages=messages_to_send,
        temperature=0,
    )
    pred_door = response.choices[0].message.content

    def normalize(coord):
        if isinstance(coord, str):
            import re
            nums = re.findall(r'\d+', coord)
            return [int(n) for n in nums]
        return [int(coord[0]), int(coord[1])]

    p_door = normalize(pred_door)
    f_door = normalize(first_door_coord)
    d_coords = [normalize(d) for d in door_coords]

    print(f"Predicted: {p_door} | Actual: {f_door}")

    # Accuracy Checks
    if p_door in d_coords:
        doorloc_acc += 1
        
    if p_door == f_door:
        dooropen_acc += 1

print(f"Door-Open Acc: {dooropen_acc / num_eval_episodes}")
print(f"Door-Location Acc: {doorloc_acc / num_eval_episodes}")