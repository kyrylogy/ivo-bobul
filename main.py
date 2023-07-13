import os
import warnings

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from architectures import *
from datasets import RandomImagePixelationDataset
from utils import plot, stack_with_padding

BATCH_SIZE = 32
def evaluate_model(model: torch.nn.Module, loader: torch.utils.data.DataLoader, loss_fn, device: torch.device):
    model.eval()
    loss = 0
    with torch.no_grad(): 
        for data in tqdm(loader, desc="Evaluating", position=0, leave=False):
 
            inputs, knowns, targets, image_files, standardization_data = data
            inputs = inputs.to(device)
            knowns = knowns.to(device)
            targets = torch.stack(targets)
            targets = targets.to(device)
            inputs = torch.cat((inputs, knowns), dim=1)
            targets = torch.cat((targets, knowns), dim=1)
            
            outputs = model(inputs)
            outputs = outputs * ~knowns
            targets = targets * ~knowns
            destandardized_outputs = outputs * standardization_data["pixelated_image"]["std"] + standardization_data["pixelated_image"]["mean"]
            destandardized_targets = targets * standardization_data["target_array"]["std"] + standardization_data["target_array"]["mean"]
            loss += torch.sqrt(loss_fn(destandardized_outputs, destandardized_targets)).item()
    loss /= len(loader)
    model.train()
    return loss


def main(
        results_path,
        network_config: dict,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        n_updates: int = 50_000,
        device: str = "cuda"
):
    device = torch.device(device)
    if "cuda" in device.type and not torch.cuda.is_available():
        warnings.warn("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    
    np.random.seed(0)
    torch.manual_seed(0)
    
    plot_path = os.path.join(results_path, "plots")
    os.makedirs(plot_path, exist_ok=True)
    
    pixelated_dataset = RandomImagePixelationDataset("data", (4,32), (4,32), (4,16), dtype=np.float32)

    training_set_size = int(0.85 * len(pixelated_dataset))
    validation_set_size = len(pixelated_dataset) - training_set_size
    training_set, validation_set = torch.utils.data.random_split(pixelated_dataset, [training_set_size, validation_set_size]) 
    
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=stack_with_padding)
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=stack_with_padding)
    
    
    net = ResidualCNN(**network_config)
    net.to(device)
    
    mse = torch.nn.MSELoss()
    
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    
    update = 0
    saved_model_file = os.path.join(results_path, "best_model.pth")
    # torch.save(net, saved_model_file)
    update_progress_bar = tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)  
    while update < n_updates:
        for data in train_loader:
            # Get single image samples
            inputs, knowns, targets, image_files, standardization_data = data
            inputs = inputs.to(device)
            knowns = knowns.to(device)
            targets = torch.stack(targets)
            targets = targets.to(device)
            inputs = torch.cat((inputs, knowns), dim=1)
            targets = torch.cat((targets, knowns), dim=1)
            
            optimizer.zero_grad()
            
            outputs = net(inputs)
            outputs = outputs * ~knowns
            targets = targets * ~knowns
            destandardized_outputs = outputs * standardization_data["pixelated_image"]["std"] + standardization_data["pixelated_image"]["mean"]
            destandardized_targets = targets * standardization_data["target_array"]["std"] + standardization_data["target_array"]["mean"]
            loss = mse(destandardized_outputs, destandardized_targets)
            loss = torch.sqrt(loss)
            loss.backward()
            optimizer.step()
            update_progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            update_progress_bar.update()
            
            
            update += 1
            if update >= n_updates:
                break
    update_progress_bar.close() 
    print("Finished Training!")
    torch.save(net.state_dict(), saved_model_file)
    
    print(f"Computing scores for best model...")
    trained_net = torch.load(saved_model_file, map_location=device)
    net.load_state_dict(trained_net)


    train_loss = evaluate_model(net, loader=train_loader, loss_fn=mse, device=device)
    val_loss = evaluate_model(net, loader=val_loader, loss_fn=mse, device=device)
    
    print(f"Scores:")
    print(f"  training loss: {train_loss}")
    print(f"validation loss: {val_loss}")
    
    with open(os.path.join(results_path, "results.txt"), "w") as rf:
        print(f"Scores:", file=rf)
        print(f"  training loss: {train_loss}", file=rf)
        print(f"validation loss: {val_loss}", file=rf)


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to JSON config file.")
    args = parser.parse_args()
    
    with open(args.config_file) as cf:
        config = json.load(cf)
    main(**config)
