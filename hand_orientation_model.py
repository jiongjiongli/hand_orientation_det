import torch
import torch.nn as nn
import torch.optim as optim


# Define the model
class HandOrientationModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(HandOrientationModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h_0 = torch.zeros(2, x.size(0), 64).to(x.device)  # Initial hidden state
        c_0 = torch.zeros(2, x.size(0), 64).to(x.device)  # Initial cell state

        # Propagate input through LSTM
        out, _ = self.lstm(x, (h_0, c_0))

        # Take the output from the last time step
        out = out[:, -1, :]

        # Pass the output through the dense layer and softmax activation
        out = self.fc(out)
        out = self.softmax(out)
        return out

# Hyperparameters
input_size = 21 * 2  # 21 key points, each with x and y coordinates
hidden_size = 64
output_size = 5  # 5 classes: up, down, clockwise, counterclockwise, unknown
num_layers = 2

# Instantiate the model, loss function, and optimizer
model = HandOrientationModel(input_size, hidden_size, output_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Model summary (optional, use torchsummary for better output)
print(model)
