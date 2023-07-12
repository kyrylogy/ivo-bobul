import os
import warnings

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from architectures import SimpleCNN
from datasets import RandomImagePixelationDataset
from utils import plot, stack_with_padding


def evaluate_model(model: torch.nn.Module, loader: torch.utils.data.DataLoader, loss_fn, device: torch.device):
    model.eval()
    loss = 0
    with torch.no_grad(): 
        for data in tqdm(loader, desc="Evaluating", position=0, leave=False):

            inputs, knowns, targets, image_files = data
            inputs = inputs.to(device)
            knowns = knowns.to(device)
            targets = torch.stack(targets)
            targets = targets.to(device)
            inputs = torch.cat((inputs, knowns), dim=1)
            
            outputs = model(inputs)
            outputs = outputs * ~knowns
            targets = targets * ~knowns
            loss += loss_fn(outputs, targets).item()
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
    
    training_set = torch.utils.data.Subset(
        pixelated_dataset,
        indices=np.arange(int(len(pixelated_dataset) * (3 / 5)))
    )
    validation_set = torch.utils.data.Subset(
        pixelated_dataset,
        indices=np.arange(int(len(pixelated_dataset) * (3 / 5)), int(len(pixelated_dataset) * (4 / 5)))
    )
    test_set = torch.utils.data.Subset(
        pixelated_dataset,
        indices=np.arange(int(len(pixelated_dataset) * (4 / 5)), len(pixelated_dataset))
    )
    
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=8, shuffle=False, num_workers=0, collate_fn=stack_with_padding)
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size=8, shuffle=False, num_workers=0, collate_fn=stack_with_padding)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=False, num_workers=0, collate_fn=stack_with_padding)
    
    writer = SummaryWriter(log_dir=os.path.join(results_path, "tensorboard"))
    
    net = SimpleCNN(**network_config)
    net.to(device)
    
    mse = torch.nn.MSELoss()
    
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    write_stats_at = 1000
    plot_at = 10_000
    validate_at = 5000 
    update = 0 
    best_validation_loss = np.inf
    update_progress_bar = tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)
    
    saved_model_file = os.path.join(results_path, "best_model.pt")
    torch.save(net, saved_model_file)
    
    while update < n_updates:
        for data in train_loader:
            # Get single image samples
            inputs, knowns, targets, image_files = data
            inputs = inputs.to(device)
            knowns = knowns.to(device)
            targets = torch.stack(targets)
            targets = targets.to(device)
            inputs = torch.cat((inputs, knowns), dim=1)
            
            optimizer.zero_grad()
            
            outputs = net(inputs)
            outputs = outputs * ~knowns
            targets = targets * ~knowns
            loss = mse(outputs, targets)
            loss.backward()
            optimizer.step()
            
            if (update + 1) % write_stats_at == 0:
                writer.add_scalar(tag="Loss/training", scalar_value=loss.cpu(), global_step=update)
                for i, (name, param) in enumerate(net.named_parameters()):
                    writer.add_histogram(tag=f"Parameters/[{i}] {name}", values=param.cpu(), global_step=update)
                    writer.add_histogram(tag=f"Gradients/[{i}] {name}", values=param.grad.cpu(), global_step=update)
            
            if (update + 1) % plot_at == 0:
                plot(inputs.detach().cpu().numpy(), targets.detach().cpu().numpy(), outputs.detach().cpu().numpy(),
                     plot_path, update)
            
            if (update + 1) % validate_at == 0:
                val_loss = evaluate_model(net, loader=val_loader, loss_fn=mse, device=device)
                writer.add_scalar(tag="Loss/validation", scalar_value=val_loss, global_step=update)
                # Save best model for early stopping
                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    torch.save(net, saved_model_file)
            
            update_progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            update_progress_bar.update()
            
            update += 1
            if update >= n_updates:
                break
    
    update_progress_bar.close()
    writer.close()
    print("Finished Training!")
    
    print(f"Computing scores for best model")
    net = torch.load(saved_model_file)
    train_loss = evaluate_model(net, loader=train_loader, loss_fn=mse, device=device)
    val_loss = evaluate_model(net, loader=val_loader, loss_fn=mse, device=device)
    test_loss = evaluate_model(net, loader=test_loader, loss_fn=mse, device=device)
    
    print(f"Scores:")
    print(f"  training loss: {train_loss}")
    print(f"validation loss: {val_loss}")
    print(f"      test loss: {test_loss}")
    
    with open(os.path.join(results_path, "results.txt"), "w") as rf:
        print(f"Scores:", file=rf)
        print(f"  training loss: {train_loss}", file=rf)
        print(f"validation loss: {val_loss}", file=rf)
        print(f"      test loss: {test_loss}", file=rf)


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to JSON config file.")
    args = parser.parse_args()
    
    with open(args.config_file) as cf:
        config = json.load(cf)
    main(**config)
