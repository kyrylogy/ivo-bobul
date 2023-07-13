import os

from matplotlib import pyplot as plt
import torch
import numpy as np


def plot(inputs, targets, predictions, path, update):
    """Plotting the inputs, targets and predictions to file ``path``."""
    os.makedirs(path, exist_ok=True)
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
    
    for i in range(len(inputs)):
        for ax, data, title in zip(axes, [inputs, targets, predictions], ["Input", "Target", "Prediction"]):
            ax.clear()
            ax.set_title(title)
            ax.imshow(data[i, 0], cmap="gray", interpolation="none")
            ax.set_axis_off()
        fig.savefig(os.path.join(path, f"{update:07d}_{i:02d}.png"), dpi=100)
    
    plt.close(fig)



def stack_with_padding(batch_as_list: list):
    # Expected list elements are 4-tuples:
    # (pixelated_image, known_array, target_array, image_file)
    n = len(batch_as_list)
    pixelated_images_dtype = batch_as_list[0][0].dtype  # Same for every sample
    known_arrays_dtype = batch_as_list[0][1].dtype
    shapes = []
    pixelated_images = []
    known_arrays = []
    target_arrays = []
    image_files = []
    
    for pixelated_image, known_array, target_array, image_file, standardization_data in batch_as_list:
        shapes.append(pixelated_image.shape)  # Equal to known_array.shape
        pixelated_images.append(pixelated_image)
        known_arrays.append(known_array)
        target_arrays.append(torch.from_numpy(target_array))
        image_files.append(image_file)
    
    max_shape = np.max(np.stack(shapes, axis=0), axis=0)
    stacked_pixelated_images = np.zeros(shape=(n, *max_shape), dtype=pixelated_images_dtype)
    stacked_known_arrays = np.ones(shape=(n, *max_shape), dtype=known_arrays_dtype)
    
    for i in range(n):
        channels, height, width = pixelated_images[i].shape
        stacked_pixelated_images[i, :channels, :height, :width] = pixelated_images[i]
        stacked_known_arrays[i, :channels, :height, :width] = known_arrays[i]
    
    return torch.from_numpy(stacked_pixelated_images), torch.from_numpy(
        stacked_known_arrays), target_arrays, image_files, standardization_data