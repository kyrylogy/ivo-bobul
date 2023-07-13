from submission_serialization import serialize, deserialize
import numpy as np
import pickle
from architectures import *
import json
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from typing import Optional
# debug_data = deserialize("targets_debug.data")

# print(debug_data[0].shape)

with open("working_config.json", "r") as fh:
    config = json.load(fh)
    network_config = config["network_config"]


class StandardizedDataset(Dataset):
    
    def __init__(
            self,
            inputs,
            knowns,
            dtype: Optional[type] = None
    ):
        self.dtype = dtype
        self.inputs = inputs
        self.knowns = knowns
    
    def __getitem__(self, index):
        image, known_array = self.inputs[index], self.knowns[index] 
        image = np.array(image, dtype=self.dtype)
        # known_array = np.array(known_array, dtype=self.dtype)
        standardization_data = {
            "inputs": {
                "mean": image.mean(),
                "std": image.std()
            },
        }
        standardized_image = (image - image.mean()) / image.std()
        return standardized_image, known_array, standardization_data
    
    def __len__(self):
        return len(self.inputs)


def stack_with_padding(batch_as_list: list):
    # Expected list elements are 4-tuples:
    # (pixelated_image, known_array, standardization_data)
    n = len(batch_as_list)
    pixelated_images_dtype = batch_as_list[0][0].dtype  # Same for every sample
    known_arrays_dtype = batch_as_list[0][1].dtype
    shapes = []
    pixelated_images = []
    known_arrays = []
    image_files = []
    
    for pixelated_image, known_array, standardization_data in batch_as_list:
        shapes.append(pixelated_image.shape)  # Equal to known_array.shape
        pixelated_images.append(pixelated_image)
        known_arrays.append(known_array)
    
    max_shape = np.max(np.stack(shapes, axis=0), axis=0)
    stacked_pixelated_images = np.zeros(shape=(n, *max_shape), dtype=pixelated_images_dtype)
    stacked_known_arrays = np.ones(shape=(n, *max_shape), dtype=known_arrays_dtype)
    
    for i in range(n):
        channels, height, width = pixelated_images[i].shape
        stacked_pixelated_images[i, :channels, :height, :width] = pixelated_images[i]
        stacked_known_arrays[i, :channels, :height, :width] = known_arrays[i]
    
    return torch.from_numpy(stacked_pixelated_images), torch.from_numpy(
        stacked_known_arrays), standardization_data



def load_pickled_dataset(path: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    with open(path, "rb") as fh:
        dataset = pickle.load(fh)
        inputs = dataset['pixelated_images']
        knowns = dataset['known_arrays']
        dataset = StandardizedDataset(inputs, knowns, dtype=np.float32)
    return dataset
        

dataset = load_pickled_dataset("test_set.pkl")
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=stack_with_padding)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResidualCNN(**network_config)
model = model.to(device)
weights = "./results/best_model.pth"
weights= torch.load(weights, map_location=device)
model.load_state_dict(weights)
np.random.seed(0)
torch.manual_seed(0)


with torch.no_grad():
    final_predictions = []
    batch = 0
    for data in loader:
        print(batch)
        # Get single image samples
        inputs, knowns, standardization_data = data
        inputs = inputs.to(device)
        knowns = knowns.to(device)
        inputs = torch.cat((inputs, knowns), dim=1)
        outputs = model(inputs)
        outputs = outputs * ~knowns
        destandardized_outputs = outputs * standardization_data["inputs"]["std"] + standardization_data["inputs"]["mean"]
        for i, pred in enumerate(destandardized_outputs):
            final_prediction = torch.masked_select(pred[0], ~knowns[i])
            final_prediction = final_prediction.detach().cpu().numpy().astype(np.uint8)
            final_predictions.append(final_prediction)
        batch += 1 

serialize(final_predictions, "predictions.bin")
    


# with torch.no_grad():
#     for pixelated_images, known_arrays in test_loader:
#         pixelated_images = pixelated_images.to(device)
#         known_arrays = known_arrays.to(device)

#         input_data = torch.cat((pixelated_images, known_arrays), dim=1)

#         outputs = model(input_data)
#         for index in range(len(outputs)):
#             flattened_prediction = torch.masked_select(outputs[index], ~known_arrays[index])
#             flattened_prediction = flattened_prediction.detach().cpu().numpy().astype(np.uint8)
#             predicted_values.append(flattened_prediction)

#             # flattened_for_plot = flatten_prediction(outputs[index], known_arrays[index])

#             # import matplotlib.pyplot as plt
#             #
#             # flattened_for_plot = flattened_for_plot.detach().cpu().numpy().squeeze()
#             # plt.imshow(flattened_for_plot, cmap='gray')
#             # plt.axis('off')
#             # plt.show()
#         batch_index += 1
#         print(f'Batch: {batch_index}')


# serialize(predicted_values, "predictions.bin")
