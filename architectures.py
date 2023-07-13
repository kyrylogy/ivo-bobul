
import torch


class SimpleCNN(torch.nn.Module):
    
    def __init__(self, n_in_channels: int = 1, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_size: int = 7):
        super().__init__()
        
        cnn = []
        for i in range(n_hidden_layers):
            cnn.append(torch.nn.Conv2d(
                in_channels=n_in_channels,
                out_channels=n_kernels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ))
            cnn.append(torch.nn.ReLU(inplace=True))
            cnn.append(torch.nn.BatchNorm2d(n_kernels))
            n_in_channels = n_kernels
        self.hidden_layers = torch.nn.Sequential(*cnn)
        
        self.output_layer = torch.nn.Conv2d(
            in_channels=n_in_channels,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
    
    def forward(self, x):
        cnn_out = self.hidden_layers(x)
        predictions = self.output_layer(cnn_out)
        predictions = torch.sigmoid(predictions) * 255.0
        return predictions

class ResidualCNN(torch.nn.Module):
    
    def __init__(self, n_in_channels: int = 1, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_size: int = 7, scale_factor: int = 2):
        super().__init__()
        
        cnn = []
        for i in range(n_hidden_layers):
            cnn.append(torch.nn.Conv2d(
                in_channels=n_in_channels,
                out_channels=n_kernels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ))
            cnn.append(torch.nn.ReLU(inplace=True))
            cnn.append(torch.nn.BatchNorm2d(n_kernels))
            cnn.append(torch.nn.Conv2d(
                in_channels=n_kernels,
                out_channels=n_kernels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ))
            cnn.append(torch.nn.ReLU(inplace=True))
            cnn.append(torch.nn.BatchNorm2d(n_kernels))
            n_in_channels = n_kernels
        self.hidden_layers = torch.nn.Sequential(*cnn)
        
        self.output_layer = torch.nn.Conv2d(
            in_channels=n_in_channels,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
    
    def forward(self, x):
        cnn_out = self.hidden_layers(x)
        predictions = self.output_layer(cnn_out) + x
        predictions = torch.sigmoid(predictions) * 255.0
        return predictions
