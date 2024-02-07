import torch
import torch.nn as nn
import yaml
import torch.optim as optim

# Define a function to create layers based on the YAML configuration
def create_layers(layer_configs):
    layers = []
    for layer_config in layer_configs:
        for i in range(layer_config['convolution']):
            for layer_type in layer_config['convolution_layers']:
                if layer_type == 'Conv2d':
                    layers.append(nn.Conv2d(layer_config['in_channels'][i],
                                      layer_config['out_channels'][i],
                                      kernel_size=layer_config['kernel_size'][0],
                                      stride=layer_config['stride'][0],
                                      padding=layer_config['padding']))
                elif layer_type == 'MaxPool2d':
                    layers.append(nn.MaxPool2d(kernel_size=layer_config['kernel_size'][1],
                                                stride=layer_config['stride'][1]))
                elif layer_type == 'ReLU':
                    layers.append(nn.ReLU())
                else:
                    raise ValueError(f"Invalid layer type: {layer_type}")

        ind = layer_config['convolution']           
        for layer_type in layer_config['fully_connected_layers']:
            if layer_type == 'Flatten':
                layers.append(nn.Flatten())
            elif layer_type == 'Linear':
                layers.append(nn.Linear(layer_config['in_channels'][ind],
                                     layer_config['out_channels'][ind]))
                ind+=1
            elif layer_type == 'ReLU':
                layers.append(nn.ReLU())
            elif layer_type == 'Sigmoid':
                layers.append(nn.Sigmoid())
            else:
                raise ValueError(f"Invalid layer type: {layer_type}")

    return layers

# Load YAML configuration
with open('model2/cnnmodified1.yml', 'r') as file:
    config = yaml.safe_load(file)

# Extract model configuration and create layers
model_config = config['architecture']['model']
layer_configs = model_config['layers']
layers = create_layers(layer_configs)

# # Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
# Instantiate the model
model = SimpleCNN(num_classes=config['hyperparameters']['num_classes'])

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = getattr(optim, config['optimizer']['type'])(
    model.parameters(),
    lr=config['optimizer']['learning_rate'],
    weight_decay=config['optimizer']['weight_decay']
)

print(model)
print(criterion)
print(optimizer)