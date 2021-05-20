import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from tqdm import tqdm

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
        k_folds(dafault=0)      |int                |Number of k_folds to be created if
                                                    |KFolds cross validation algorithm
                                                    |will be performed.
        model                   |torch.nn.Module    |Model class to be trained.
        (default=RestaurantsDNN)
        optimizer(default=adam) |torch.optim/list   |Torch optimizer or optimizers to use
                                                    |while training. If multiple given,
                                                    |one training will be made for each.
        loss_function           |torch.nn           |Torch loss function to be used while
        (default=BCELoss)                           |training.
        lr(default=0.01)        |float/list         |Learning rate (η) used on optimizer
                                                    |update step algorithm. If multiple
                                                    |given, one training will be done
                                                    |for each on each optimizer given.
        epochs(default=30)      |int/list           |Number of epochs used on training
                                                    |loop. If multiple given, one training
                                                    |will be done for each.
        scheduler(optional)     |torch.optim.       |Learning rate scheduler used while
                                |lr_scheduler       |training model.
        categories(optional)    |int                |number of categories to be predicted
        -------------------------------------------------------------------------------------
        Output:
        Name                    |Type               |Description

        models                  |torch.nn.Module    |List with each trained model for given
                                                    |training parameters.
        -------------------------------------------------------------------------------------
        """
        ####### VALIDATIONS #######
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
        if split_at > 1:
            raise ValueError("split_at must be a number between 0 and 1")
        self.split_at = split_at


        # Get if KFold cross validations is required
        self.k = self.validateData(kwargs, "k_folds", None)

        # model validation
        self.model = self.validateData(kwargs, "model", RestaurantsDNN)

        ### optimizer validation
        self.optimizers = self.validateData(kwargs, "optimizers", optim.Adam)
        # If only one given, make it into a list
        if not hasattr(self.optimizers, '__iter__'):
            self.optimizers = [self.optimizers]

        ### loss function validation
        self.lossFunction = self.validateData(kwargs, "loss_function", nn.BCELoss)

        ### learning rate validation
        self.lr = self.validateData(kwargs, "lr", 0.01)
        # If only one given, make it into a list
        if not hasattr(self.lr, '__iter__'):
            self.lr = [self.lr]

        ### epochs validation
        self.epochs = self.validateData(kwargs, "epochs", 30)
        # If only one given, make it into a list
        if not hasattr(self.epochs, '__iter__'):
            self.epochs = [self.epochs]

        ### scheduler validation
        self.scheduler = self.validateData(kwargs, "scheduler", None)

        ### categories validation
        self.categories = self.validateData(kwargs, "categories", None)
        if not self.categories:
            raise ValueError("Yo need at least two categories to be predicted")

        ####### DATA PREPARATION ######
        self.random_state = random_state
        # Get x and y
        self.xtrn = None; self.xtst = None; self.ytrn = None; self.ytst = None;
        self._splitData() # split data

        ###### TRAINING ########
        for opt in tqdm(self.optimizers):
            for lr in tqdm(self.lr):
                model = self.model(self.categories) if self.model == RestaurantsDNN else self.model()
                optim = opt(model.parameters(), lr=lr)
                model = ModelTraining(model, optimizer=optim, lossFunction=self.lossFunction,
                            epochs=self.epochs, η=self.lr, k_folds=self.k, categories=self.categories)
                if self.scheduler:
                    scheduler = self.scheduler(optim, 'min', verbose=True)
                    model.scheduler = scheduler

                trainedModel = model.trainModel(self.xtrn, self.ytrn)
                modelPerformance = ModelPerformance(trainedModel, self.categories)
                print(f"Performance over training set is: {modelPerformance.modelEvaluation(self.xtrn, self.ytrn)}")
                print(f"Performance over test set is: {modelPerformance.modelEvaluation(self.xtst, self.ytst)}")




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
        if self.random_state:
            xTrn, xTst, yTrn, yTst = train_test_split(x, y, test_size=split_at, random_state=self.random_state)
        else:
            xTrn, xTst, yTrn, yTst = train_test_split(x, y, test_size=split_at)

        # Convert data into torch tensors as floats
        self.xtrn = torch.tensor(x_train).type(torch.FloatTensor); self.xtst = torch.tensor(x_test).type(torch.FloatTensor);
        self.ytrn = torch.tensor(y_train).type(torch.FloatTensor); self.ytst = torch.tensor(y_test).type(torch.FloatTensor);

    def validateData(self, kwargs, key, default):
        val = None
        try:
            val = kwargs[key]
        except KeyError:
            val = dafault
            pass
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise
        return val


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
    def __init__(self, model, optimizer=None, lossFunction=None, scheduler=None, epochs=30, η=0.01, k_folds=False, categories=2):
        self.epochs = epochs
        self.η = η
        self.optim = optim.SGD()
        self.lossFunction = nn.BCELoss()
        self.k = k_folds
        self.model = model
        self.categories = categories
        self.scheduler = scheduler

        if self.k_folds:
            kf = KFold(n_splits=self.k)
            self.k_folds = kf.get_n_splits(x).split(x, y)
        else:
            if self.scheduler:
                print("WARNING: Scheduler will be ignored beacuse there's no k_folds value given")

    def trainModel(self, x, y):
        print(f"Training model for {self.epochs} epochs...")
        for epoch in tqdm(range(self.epochs)):
            if self.k:
                trnInd, valInd = self.k_folds[epochs % k]
                xTrn = x[trnInd]; yTrn = y[trnInd];
                xVal = x[valInd]; yVal = y[valInd];
            else:
                xTrn = x; yTrn = y;
            self.optim.zero_grad()
            pred = self.model(xTrn)
            pred = pred.view(pred.size()[0]) # Reshape predictions
            loss = self.lossFunction(pred, x)
            loss.backward() # Backpropagate
            self.optim.step() # Update params
            acc = self.modelAccuracy(yTrn, pred)
            #print(f"Trn set:\nepoch: {epoch} loss: {loss} Acc: {acc}"
            if self.scheduler and self.k:
                pred = self.model(xVal)
                pred = pred.view(pred.size()[0]) # Reshape predictions
                acc = self.modelAccuracy(yVal, pred)
                #print(f"Val set:\nepoch: {epoch} loss: {loss} Acc: {acc}"
                self.scheduler.step(loss.item())
        performance = ModelPerformance(self.model, self.categories)
        print(f"Trained model performance :\n{performance}")
        return self.model


    def modelAccuracy(self, y, pred):
        acc = sum(torch.round(pred * self.categories) / self.categories == y) # perform accuracy
        return acc /= y.size()[0]
