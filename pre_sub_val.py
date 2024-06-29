# imports
import torch
import yaml
import time
from model import Model
from utils.chess_gameplay import Agent, play_game

# model instantiation
model_config = yaml.safe_load(open("model_config.yaml"))
model0 = Model(**model_config)

# checkpoint loading
checkpoint = torch.load("checkpoint.pt", map_location=torch.device('cpu'))
model0.load_state_dict(checkpoint["model"])

# model inference
batch_size = torch.randint(2, 50, (1,))
inputs = torch.randint(0, 21, (batch_size, 8, 8))
outputs = model0(inputs)

# outputs validation
assert outputs.shape == torch.Size([batch_size]), "Wrong output shape."
assert outputs.dtype == torch.float32, "Wrong output data type."
assert outputs.device == torch.device(type='cpu'), "Outputs on wrong device."
print("Outputs pass validation tests.")

# testing gameplay
model1 = Model(**model_config)
agent0, agent1 = Agent(model0), Agent(model1)
gameplay_kwargs = {
    "table": 1,
    "agents": {'white': agent0, 'black': agent1},
    "max_moves": 50,
    "min_seconds_per_move": 0.0,
    "verbose": False,
    "poseval": False
}

timer_start = time.perf_counter()
game_result = play_game(**gameplay_kwargs)
elapsed = time.perf_counter() - timer_start
assert elapsed < 200, "Model too slow, consider simplifying or reducing the size of your model."
print("Model passes validation test.")