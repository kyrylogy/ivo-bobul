import torch
from architectures import SimpleCNN
from datasets import RandomImagePixelationDataset
from utils import stack_with_padding
import json
import numpy as np
import matplotlib.pyplot as plt

pixelated_dataset = RandomImagePixelationDataset("data", (4,32), (4,32), (4,16), dtype=np.float32)

validation_set = torch.utils.data.Subset(
    pixelated_dataset,
    indices=np.arange(int(len(pixelated_dataset) * (3 / 5)), int(len(pixelated_dataset) * (4 / 5)))
)

val_loader = torch.utils.data.DataLoader(validation_set, batch_size=8, shuffle=False, num_workers=0, collate_fn=stack_with_padding)
device = torch.device("cuda")
if "cuda" in device.type and not torch.cuda.is_available():
    device = torch.device("cpu")
    

with open("working_config.json", "r") as f:
    full_config = json.load(f)
    network_config = full_config["network_config"]


model_path = "results/best_model.pt"
model = SimpleCNN(**network_config)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
np.random.seed(0)
torch.manual_seed(0)



for data in val_loader:
    # Get single image samples
    inputs, knowns, targets, image_files, standardization_data = data
    inputs = inputs.to(device)
    knowns = knowns.to(device)
    targets = torch.stack(targets)
    targets = targets.to(device)
    inputs = torch.cat((inputs, knowns), dim=1)
    
    outputs = model(inputs)
    outputs = outputs * ~knowns
    targets = targets * ~knowns
    destandardized_outputs = outputs * standardization_data["pixelated_image"]["std"] + standardization_data["pixelated_image"]["mean"]
    destandardized_targets = targets * standardization_data["target_array"]["std"] + standardization_data["target_array"]["mean"]


    # plot target
    plt.imshow(destandardized_targets[0].cpu().numpy(), cmap="gray")
