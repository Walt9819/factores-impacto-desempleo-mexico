import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


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
        return f"Restaurants DNN with {self.inputNeurons} input neurons"


class RestaurantsModel():
    def __init__(self, dataset=None, split_at=0.3, random_state=None, **kwargs):
        # Get x and y from given params
        if not dataset:
            try:
                x = kwargs["x"]
                y = kwargs["y"]
            except KeyError as e:
                raise ValueError("x and/or y datasets not given")
        else:
            x = dataset["x"]
            y = dataset["y"]

        if split_at > 1:
            raise ValueError("split_at must be a number between 0 and 1")

        # Split datasets into train and test
        xTrn, xTst, yTrn, yTst = train_test_split(x, y, test_size=split_at, random_state=random_state)

        # Convert data into torch tensors as floats
        xtrn = torch.tensor(x_train).type(torch.FloatTensor); xtst = torch.tensor(x_test).type(torch.FloatTensor);
        ytrn = torch.tensor(y_train).type(torch.FloatTensor); ytst = torch.tensor(y_test).type(torch.FloatTensor);
