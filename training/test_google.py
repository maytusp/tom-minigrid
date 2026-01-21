import google.generativeai as genai
import os
from .load_obs import get_input_output

try:
    with open("/home/maytus/tom/test_llm/mykey.txt", "r") as f:
        api_key = f.read().strip() 
except FileNotFoundError:
    print("Error: 'mykey.txt' not found. Please create the file with your API key.")
    exit()

genai.configure(api_key=api_key)
# print("Listing available models...")
# for m in genai.list_models():
#     if 'generateContent' in m.supported_generation_methods:
#         print(m.name)



# Initialize the model (Gemini 2.0 or 1.5 are current powerful versions)
model = genai.GenerativeModel('models/gemini-2.5-flash')

static_instruction = """
### ROLE & CONTEXT
You are an observer of a protagonist agent (P) in a symbolic 9x9 grid world. 
The agent is trained via RL to find the STAR (12) and reach the GOAL (6). After the star is reached, it will disappear.

### GRID DEFINITIONS
The grid is 0-indexed:
- Top-left is (0,0).
- Bottom-right is (8,8).
- Format is (row, column).

### LEGEND (Value_dict)
0: EMPTY, 1: FLOOR, 2: WALL, 3: BALL, 4: SQUARE, 
5: PYRAMID, 6: GOAL, 7: KEY, 8: DOOR_LOCKED, 
9: DOOR_CLOSED, 10: DOOR_OPEN, 11: HEX, 12: STAR

### AGENT IDENTIFIERS
The Agent (P) is strictly defined by the following values (depending on orientation):
- 13 (AGENT_UP)
- 14 (AGENT_RIGHT)
- 15 (AGENT_DOWN)
- 16 (AGENT_LEFT)

### TASK
Predict what door will P open first steps after reaching a star.
Return the coordinate of the cell containing the opened door (value = 10).

### INPUT FORMAT
A part of episode represented by a sequence of symbolic observation.
This part ends when P finds the star.

### OUTPUT FORMAT
Provide ONLY the coordinate tuple in brackets. Do not write explanations.
Example: (2, 3)
"""



file_path = "logs/trajs/MiniGrid-ToM-TwoRoomsSwap-9x9vs9/ep_005_observer_sym.npz"
total_observations, seg_obs, star_idx, first_door_coord = get_input_output(file_path)
observations_data = seg_obs
dynamic_log = "Below is the observation log:\n"

for i, obs in enumerate(observations_data):
    # Format the grid nicely so the model understands the 2D structure
    formatted_grid = "\n".join([str(row) for row in obs])
    dynamic_log += f"\n--- Time Step {i} ---\n{formatted_grid}\n"

final_prompt = static_instruction + dynamic_log
response = model.generate_content(final_prompt)
print(response.text)

print(f"Groundtruth :{first_door_coord}")
# TB
# "logs/trajs/MiniGrid-ToM-TwoRoomsNoSwap-9x9vs9/ep_000_observer_sym.npz", (2,1), wrong door
# "logs/trajs/MiniGrid-ToM-TwoRoomsNoSwap-9x9vs9/ep_001_observer_sym.npz", (1,6), correct
# "logs/trajs/MiniGrid-ToM-TwoRoomsNoSwap-9x9vs9/ep_002_observer_sym.npz", (4,7), correct
# "logs/trajs/MiniGrid-ToM-TwoRoomsNoSwap-9x9vs9/ep_003_observer_sym.npz", (5,7), wrong door
# "logs/trajs/MiniGrid-ToM-TwoRoomsNoSwap-9x9vs9/ep_004_observer_sym.npz", (7,6), wrong door


# FB
# file_path = "logs/trajs/MiniGrid-ToM-TwoRoomsSwap-9x9vs9/ep_000_observer_sym.npz", (2,1), wrong door
# file_path = "logs/trajs/MiniGrid-ToM-TwoRoomsSwap-9x9vs9/ep_001_observer_sym.npz", (6,1), wrong door
# file_path = "logs/trajs/MiniGrid-ToM-TwoRoomsSwap-9x9vs9/ep_002_observer_sym.npz", (4,7), correct
# file_path = "logs/trajs/MiniGrid-ToM-TwoRoomsSwap-9x9vs9/ep_003_observer_sym.npz", (5,7), wrong door
# file_path = "logs/trajs/MiniGrid-ToM-TwoRoomsSwap-9x9vs9/ep_005_observer_sym.npz", (5,7), correct
