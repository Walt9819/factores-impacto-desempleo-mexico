import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

from tqdm import tqdm

import torch.nn as nn


class RestaurantsDNN(nn.Module):
    def __init__(self, inputNeurons):
        super().__init__()
        self.relu = nn.ReLU()
        self.inputNeurons = inputNeurons
        self.input = nn.Linear(inputNeurons, 100)
        self.f1 = nn.Linear(100, 150)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.3) # dropput layer with 0.3 prob
        self.f2 = nn.Linear(150, 50)
        self.output = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.tanh(self.input(x))
        x = self.relu(self.f1(x))
        x = self.dropout(x)
        x = self.relu(self.f2(x))
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

        import torch.optim as optim

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
            #if (self.categories := len(torch.unique(y))) > 2: # might work on python 3.8
            self.categories = len(np.unique(y))
            if not self.categories >= 2:
                raise ValueError("You need at least two categories to be predicted")

        ####### DATA PREPARATION ######
        self.random_state = random_state
        # Get x and y
        self.xtrn = None; self.xtst = None; self.ytrn = None; self.ytst = None;
        self._splitData(x, y) # split data


    def trainModels(self):
        """
        Train models from given params inside instance.
        -------------------------------------------------------------------------------------
        Params:
        Name                    |Type               |Description
        -------------------------------------------------------------------------------------
        Output:
        Name                    |Type               |Description
        -------------------------------------------------------------------------------------
        """
        models = []
        for opt in tqdm(self.optimizers):
            for lr in tqdm(self.lr):
                for epoch in tqdm(self.epochs):
                    model = self.model(self.xtrn.size()[1]) if self.model == RestaurantsDNN else self.model()
                    optim = opt(model.parameters(), lr=lr)
                    model = ModelTraining(model, optimizer=opt, lossFunction=self.lossFunction,
                                epochs=epoch, η=lr, k_folds=self.k, categories=self.categories)
                    if self.scheduler:
                        scheduler = self.scheduler(optim, 'min', verbose=True)
                        model.scheduler = scheduler

                    trainedModel = model.trainModel(self.xtrn, self.ytrn)
                    models.append(trainedModel)
                    modelPerformance = ModelPerformance(trainedModel, self.categories)
                    modelPerformance.modelEvaluation(self.xtrn, self.ytrn)
                    print(f"Performance over training set is:\n")
                    modelPerformance.display()
                    modelPerformance.modelEvaluation(self.xtst, self.ytst)
                    print(f"Performance over test set is:")
                    modelPerformance.display()
        return models



    def _splitData(self, x, y):
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
            xTrn, xTst, yTrn, yTst = train_test_split(x, y, test_size=self.split_at, random_state=self.random_state)
        else:
            xTrn, xTst, yTrn, yTst = train_test_split(x, y, test_size=self.split_at)

        # Convert data into torch tensors as floats
        self.xtrn = torch.tensor(xTrn).type(torch.FloatTensor); self.xtst = torch.tensor(xTst).type(torch.FloatTensor);
        self.ytrn = torch.tensor(yTrn).type(torch.FloatTensor); self.ytst = torch.tensor(yTst).type(torch.FloatTensor);

    def validateData(self, kwargs, key, default):
        val = None
        try:
            val = kwargs[key]
        except KeyError:
            val = default
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
        self.evals = {}


    def modelEvaluation(self, x, y, matrix_plot=False):
        """
        Evaluate model performance for given data
        -------------------------------------------------------------------------------------
        Params:
        Name                    |Type               |Description
        x                       |iterable           |Input data.
        y                       |iterable           |Labels data.
        matrix_plot             |bool               |Flag if confussion matrix will be ploted.
        labels (optional)       |iterable           |Iterable with labels to be ploted.
        -------------------------------------------------------------------------------------
        Output:
        Name                    |Type               |Description
        evals                   |Dict               |Dictionary with keys for each available
                                                    |performance test (accuracy), if
                                                    |`categories` is equal to 2, more tests given
                                                    |(sensitivity and specificity).
        -------------------------------------------------------------------------------------
        """
        if x.size()[0] != y.size()[0]:
            raise ValueError(f"Mismatched dimensions: input dimension is {x.size()[0]} and labels input is {y.size()[0]}")
        with torch.no_grad():
            sizeY = y.size()[0]
            pred = self.model(x) # predictions
            pred = pred.view(pred.size()[0]) # Reshape predictions
            acc = sum(torch.round(pred * self.categories) / self.categories == y) # perform accuracy
            self.evals["Accuracy"] = acc / sizeY
            if self.categories == 2:
                TPRes = torch.round(pred) * y
                TP = sum(TPRes) # ∑ (yŷ)
                FP = abs(sum(TPRes - pred)) # |∑ (yŷ - ̂y)|
                FN = abs(sum(TPRes - y)) # |∑ (yŷ - y)|
                TN = sizeY - FP + FN + TP # sum(y + ̂y - TP) - sizeY
                self.evals["Sensitivity"] = TP / (TP + FN)
                self.evals["Specificity"] = TN / (TN + FP)
            if matrix_plot:
                confMat = [[0 for i in range(self.categories)] for j in range(self.categories)]
                for i in range(sizeY):
                    confMat[pr[i]][y[i]] += 1
                self.plotConfMatrix(confMat) if not labels else self.plotConfMatrix(confMat, labels)
        return self.evals


    def plotConfMatrix(self, confMatrix, labels=None):
        import seaborn as sns
        if labels:
            x_axis_labels = labels # labels for x-axis
            y_axis_labels = labels # labels for y-axis
            ax = sns.heatmap(confMatrix, cmap="Blues", annot=True, fmt="d", xticklabels=x_axis_labels, yticklabels=y_axis_labels)
        else:
            ax = sns.heatmap(confMatrix, cmap="Blues", annot=True, fmt="d")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion matrix");
        plt.show()


    def display(self):
        for key, val in self.evals.items():
            print("{}\t\t:\t\t{:.4f}".format(key, val))



class ModelTraining():
    def __init__(self, model, optimizer=None, lossFunction=None, scheduler=None, epochs=30, η=0.01, k_folds=False, categories=2):
        self.epochs = epochs
        self.η = η
        self.model = model
        self.optim = optim.SGD(model.parameters(), lr=self.η)
        self.lossFunction = nn.BCELoss()
        self.k = k_folds
        self.categories = categories
        self.scheduler = scheduler

    def trainModel(self, x, y):
        if self.k:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=self.k)
            k_folds = kf.split(x, y)
        else:
            if self.scheduler:
                print("WARNING: Scheduler will be ignored beacuse there's no k_folds value given")
        print(f"\nTraining model for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            if self.k:
                index = next(iter(k_folds), None)
                if index == None:
                  k_folds = KFold(n_splits=self.k).split(x, y)
                  index = next(iter(k_folds), None)
                trnInd, valInd = index
                xTrn = x[trnInd]; yTrn = y[trnInd];
                xVal = x[valInd]; yVal = y[valInd];
            else:
                xTrn = x; yTrn = y;
            self.optim.zero_grad()
            pred = self.model(xTrn)
            pred = pred.view(pred.size()[0]) # Reshape predictions
            loss = self.lossFunction(pred, yTrn)
            loss.backward() # Backpropagate
            self.optim.step() # Update params
            acc = self.modelAccuracy(yTrn, pred)
            if epoch % 100 == 0:
              print(f"Trn set:\nEpoch: {epoch} Loss: {loss:.4f} Acc: {acc:.4f}")
            if self.k:
                pred = self.model(xVal)
                pred = pred.view(pred.size()[0]) # Reshape predictions
                acc = self.modelAccuracy(yVal, pred)
                if epoch % 100 == 0:
                  print(f"Val set:\nEpoch: {epoch} loss: {loss:.4f} Acc: {acc:.4f}")
                if self.scheduler:
                  self.scheduler.step(loss.item())
        modelEvaluator = ModelPerformance(self.model, self.categories)
        performance = modelEvaluator.modelEvaluation(xTrn, yTrn)
        print(f"\nTrained model performance :")
        modelEvaluator.display()
        return self.model


    def modelAccuracy(self, y, pred):
        acc = sum(torch.round(pred * self.categories) / self.categories == y) # perform accuracy
        return acc / y.size()[0]
