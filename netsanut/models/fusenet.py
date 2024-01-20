import torch 
import torch.nn as nn 

class FuseNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fwd_module = nn.Linear(10, 1)
    
    def forward(self, data_dict):
        
        print("FuseNet forward")
        
        loss_dict = {}
        
        return 

    def adapt_to_metadata(self, metadata):
        
        pass 
    
def build_model(cfg):
    
    print("build FuseNet")
    
    return FuseNet()


import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class TimeSeriesFusionModelGNN(nn.Module):
    
    """ This is from ChatGPT, just for a reference
    """
    
    def __init__(self, input_shape_sensor1, input_shape_sensor2, adjacency_matrix, output_dim):
        super(TimeSeriesFusionModelGNN, self).__init__()

        # Sensor 1 processing
        self.conv1_sensor1 = nn.Conv2d(in_channels=input_shape_sensor1[2], out_channels=64, kernel_size=(3, 3))
        self.lstm1_sensor1 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)

        # Sensor 2 processing
        self.conv1_sensor2 = nn.Conv2d(in_channels=input_shape_sensor2[2], out_channels=64, kernel_size=(3, 3))
        self.lstm1_sensor2 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)

        # Fusion at the time dimension
        self.concatenated_time = nn.Sequential()

        # Graph convolutional layer for spatial processing
        num_nodes = position  # Number of nodes in the graph (positions)
        num_features = 64  # Number of features for each node after time processing
        self.gcn = GCNConv(num_features, num_features)

        # Spatial processing
        self.conv2 = nn.Conv2d(in_channels=num_features, out_channels=128, kernel_size=(3, 3))
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(128 * (input_shape_sensor1[1] + input_shape_sensor2[1] - 2), 256)

        # Output layer
        self.output_layer = nn.Linear(256, output_dim)

    def forward(self, x_sensor1, x_sensor2, adjacency_matrix):
        # Sensor 1 processing
        x_sensor1 = self.conv1_sensor1(x_sensor1)
        _, (x_sensor1, _) = self.lstm1_sensor1(x_sensor1)

        # Sensor 2 processing
        x_sensor2 = self.conv1_sensor2(x_sensor2)
        _, (x_sensor2, _) = self.lstm1_sensor2(x_sensor2)

        # Fusion at the time dimension
        x = torch.cat([x_sensor1, x_sensor2], dim=1)

        # Graph convolutional layer for spatial processing
        x = self.gcn(x, adjacency_matrix)

        # Spatial processing
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense(x)

        # Output layer
        output = self.output_layer(x)

        return output

# Example usage:
batch_size = 32  # Replace with your desired batch size
time_steps_sensor1 = 50  # Replace with the number of time steps for sensor 1
time_steps_sensor2 = 30  # Replace with the number of time steps for sensor 2
position = 64  # Replace with the number of positions
value = 1  # Replace with the number of values at each position
output_dim = 1  # Replace with the desired output dimension

# Assuming adjacency_matrix is a torch.Tensor representing the spatial adjacency matrix
adjacency_matrix = torch.ones((position, position), dtype=torch.float32)

model_gnn = TimeSeriesFusionModelGNN(
    input_shape_sensor1=(batch_size, time_steps_sensor1, position, value),
    input_shape_sensor2=(batch_size, time_steps_sensor2, position, value),
    adjacency_matrix=adjacency_matrix,
    output_dim=output_dim
)

# Print the model architecture
print(model_gnn)


# some initial test codes
if __name__ == "__main__":
    pass
