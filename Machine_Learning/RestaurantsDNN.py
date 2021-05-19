import torch
import torch.nn as nn
import torch.optim as optim

class RestaurantsDNN(nn.Module):
    def __init__(self, inputNeurons):
        super().__init__()
        self.inputNeurons = inputNeurons
        self.input = nn.Linear(inputNeurons, 15)
        self.f1 = nn.Linear(15, 10)
        self.f2 = nn.Linear(10, 5)
        self.output = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.tanh(self.input(x))
        x = self.tanh(self.f1(x))
        x = self.dropout(x)
        x = self.tanh(self.f2(x))
        x = self.dropout(x)
        x = self.output(x)
        return self.sigmoid(x)

    def __str__(self):
        return f"RestaurantsDNN for {self.inputNeurons} input neurons"
