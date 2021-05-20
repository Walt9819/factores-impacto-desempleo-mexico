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
        self.dropout = nn.Dropout(0.25) # dropput layer with 0.25 prob

    def forward(self, x):
        x = self.tanh(self.input(x))
        x = self.tanh(self.f1(x))
        x = self.dropout(x)
        x = self.tanh(self.f2(x))
        x = self.dropout(x)
        x = self.output(x)
        return self.sigmoid(x) # probability activation layer (softmax like)

    def __str__(self):
        return f"Restaurants DNN with {self.inputNeurons} input neurons"



class RestaurantsModel():
    def __init__(self, dataset=None, split_at=0.3, random_state=None, **kwargs):
        # Get x and y from given params
        self.split_at = split_at
        self.random_state = random_state
        self.xtrn = None; self.xtst = None; self.ytrn = None; xelf.ytst = None;
        if not dataset:
            try:
                x = kwargs["x"]
                y = kwargs["y"]
            except KeyError as e:
                raise ValueError("x and/or y datasets not given")
        else:
            x = dataset["x"]
            y = dataset["y"]

        self.splitData()


    def splitData(self, x, y):
        if self.split_at > 1:
            raise ValueError("split_at must be a number between 0 and 1")

        # Split datasets into train and test
        xTrn, xTst, yTrn, yTst = train_test_split(x, y, test_size=split_at, random_state=random_state)

        # Convert data into torch tensors as floats
        self.xtrn = torch.tensor(x_train).type(torch.FloatTensor); self.xtst = torch.tensor(x_test).type(torch.FloatTensor);
        self.ytrn = torch.tensor(y_train).type(torch.FloatTensor); self.ytst = torch.tensor(y_test).type(torch.FloatTensor);



class RestaurantsTraining():
    def __init__(self, model, epochs=30, η=0.01, optimizer=optim.SGD(), lossFunction=nn.BCELoss, k_folds=False, categories=2):
        self.epochs = epochs
        self.η = η
        self.optim = optimizer
        self.lossFunction = lossFunction
        self.k_folds = k_folds
        self.model = model
        self.categories = categories


    def performAccuracy(self, y, pred):
        acc = sum(torch.round(pred * categories) / self.categories == y) # perform accuracy
        return acc


    def trainModel(self, x, y):
        print(f"Training model for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            self.optim.zero_grad()
            pred = self.model(x)
            pred = pred.view(pred.size()[0]) # Reshape predictions
            loss = self.lossFunction(pred, x)
            loss.backward() # Backpropagate
            self.optim.step() # Update params
            acc = self.performAccuracy(pred, y)
            #print(f"epoch: {epoch} loss: {loss} Acc: {acc}"
        print("Training performance:")
        self.testModel(x, y)


    def testModel(self, x, y):
        with torch.no_grad():
            testPred = self.model(x)
            testPred = testPred.view(testPred.size()[0])
            loss = self.lossFunction(pred, objetivoTrain)
            acc = performAccuracy(pred, y)
            print(f"epoch: {epoch} loss: {loss} Acc: {acc}")
