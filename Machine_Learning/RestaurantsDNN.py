import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


class RestaurantsDNN(nn.Module):
    def __init__(self, inputNeurons):
        super().__init__()
        self.relu = nn.ReLU()
        self.inputNeurons = inputNeurons
        self.input = nn.Linear(inputNeurons, 15)
        self.f1 = nn.Linear(15, 10)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.25) # dropput layer with 0.25 prob
        self.f2 = nn.Linear(10, 5)
        self.output = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.tanh(self.input(x))
        x = self.tanh(self.f1(x))
        x = self.dropout(x)
        x = self.tanh(self.f2(x))
        x = self.dropout(x)
        x = self.output(x)
        return self.sigmoid(x) # probability activation layer (softmax like)



class MainModel():
    def __init__(self, dataset=None, split_at=0.3, random_state=None, **kwargs):
        """
        Main restaurants model class init method
        -------------------------------------------------------------------------------------
        Params:
        Name                    |Type               |Description

        dataset(optional)       |Dict               |Dictionary with `x` and `y` keys for
                                                    |input and label data of iterable type.
        split_at(default=0.3)   |float              |Size of test set where to split data.
        random_state(optional)  |int                |Random state for reproducibility.
        x(optional)             |iterable           |**Only used if `dataset` not given**.
                                                    |Input data.
        y(optional)             |iterable           |**Only used if `dataset` not given**.
                                                    |Labels data.
        optimizer(default=adam) |torch.optim/list   |Torch optimizer or optimizers to use
                                                    |while training. If multiple given,
                                                    |one training will be made for each.
        lr(default=0.0.1)       |float/list         |Learning rate (η) used on optimizer
                                                    |update step algorithm. If multiple
                                                    |given, one training will be done
                                                    |for each on each optimizer given.
        scheduler(optional)     |torch.optim.       |Learning rate scheduler used while
                                |lr_scheduler       |training model.
        -------------------------------------------------------------------------------------
        Output:
        Name                    |Type               |Description

        models                  |torch.nn.Module    |List with each trained model for given
                                                    |training parameters.
        -------------------------------------------------------------------------------------
        """
        # Get x and y from given params
        self.split_at = split_at
        self.random_state = random_state
        self.xtrn = None; self.xtst = None; self.ytrn = None; xelf.ytst = None;
        # Dataset validation
        if not dataset:
            try:
                x = kwargs["x"]
                y = kwargs["y"]
            except KeyError as e:
                raise ValueError("x and/or y datasets not given")
        else:
            x = dataset["x"]
            y = dataset["y"]
        # Split validation
        if self.split_at > 1:
            raise ValueError("split_at must be a number between 0 and 1")

        # Get if k fold cross validations is required
        try:
            k = kwargs["k_folds"]
        except KeyError:
            k = None
            pass
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

        self.splitData() # split data


    def _splitData(self):
        """
        Split given data into train and test set (**read only**) method.
        -------------------------------------------------------------------------------------
        Params:
        Name                    |Type               |Description
        -------------------------------------------------------------------------------------
        Output:
        Name                    |Type               |Description
        -------------------------------------------------------------------------------------
        """
        # Split datasets into train and test
        xTrn, xTst, yTrn, yTst = train_test_split(x, y, test_size=split_at, random_state=random_state)

        # Convert data into torch tensors as floats
        self.xtrn = torch.tensor(x_train).type(torch.FloatTensor); self.xtst = torch.tensor(x_test).type(torch.FloatTensor);
        self.ytrn = torch.tensor(y_train).type(torch.FloatTensor); self.ytst = torch.tensor(y_test).type(torch.FloatTensor);


class ModelPerformance():
    def __init__(self, model, categories=2):
        """
        Evaluate model performance for given data
        -------------------------------------------------------------------------------------
        Params:
        Name                    |Type               |Description

        model                   |torch.nn.Module    |Model to be tested.
        categories(default=2)   |iterable           |Number of categories to be predicted.
        -------------------------------------------------------------------------------------
        Output:
        Name                    |Type               |Description
        -------------------------------------------------------------------------------------
        """
        self.model = model
        self.categories = categories


    def modelEvaluation(self, x, y):
        """
        Evaluate model performance for given data
        -------------------------------------------------------------------------------------
        Params:
        Name                    |Type               |Description

        x                       |iterable           |Input data.
        y                       |iterable           |Labels data.
        -------------------------------------------------------------------------------------
        Output:
        Name                    |Type               |Description

        evals                   |Dict               |Dictionary with keys for each available
                                                    |performance test ("acc": accuracy), if
                                                    |`categories` is equal to 2, more tests given
                                                    |("sens": sensitivity, "spec": specificity).
        -------------------------------------------------------------------------------------
        """
        if x.size()[0] != y.size()[0]:
            raise ValueError(f"Mismatched dimensions: input dimension is {x.size()[0]} and labels input is {y.size()[0]}")
        evals = Dict()
        with torch.no_grads():
            sizeY = y.size()[0]
            pred = self.model(x) # predictions
            pred = pred.view(pred.size()[0]) # Reshape predictions
            acc = sum(torch.round(pred * self.categories) / self.categories == y) # perform accuracy
            evals["acc"] = acc / sizeY
            if self.categories == 2:
                TPRes = torch.round(pred) * y
                TP = sum(TPRes) # ∑ (yŷ)
                FP = abs(sum(TPRes - pred)) # |∑ (yŷ - ̂y)|
                FN = abs(sum(TPRes - y)) # |∑ (yŷ - y)|
                TN = sizeY - FP + FN + TP # sum(y + ̂y - TP) - sizeY
                evals["sens"] = TP / (TP + TN)
                evals["spec"] = TN / (TN + FP)
        return evals


class ModelTraining():
    def __init__(self, model, optimizer=None, lossFunction=None, epochs=30, η=0.01, k_folds=False, categories=2):
        self.epochs = epochs
        self.η = η
        self.optim = optim.SGD()
        self.lossFunction = nn.BCELoss()
        self.k_folds = k_folds
        self.model = model
        self.categories = categories


    def trainModel(self, x, y):
        print(f"Training model for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            self.optim.zero_grad()
            pred = self.model(x)
            pred = pred.view(pred.size()[0]) # Reshape predictions
            loss = self.lossFunction(pred, x)
            loss.backward() # Backpropagate
            self.optim.step() # Update params
            acc = model.modelAccuracy(y, pred)
            #print(f"epoch: {epoch} loss: {loss} Acc: {acc}"
        performance = ModelPerformance(self.model, self.categories)
        print(f"Trained net performance is:\n{performance}")
        return self.model


    def modelAccuracy(self, y, pred):
        acc = sum(torch.round(pred * self.categories) / self.categories == y) # perform accuracy
        return acc /= y.size()[0]
