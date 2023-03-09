# cs6910_assignment1

The description of the different files in the repository are as follows:

Firstly, we have used the train.py to perform the sweep using some of the commented sections of the code.
The code written to plot onfusion matrix is also commented in the end of train.py
The sweep.yaml has the code to initialize the sweep with the required hyperparameters.
Sample Images.png has the one sample image from each of the 10 classes.
Best possible set of hyperparameters found on fashion-mnist dataset-

Activation - ReLU, 
Batch_size-32,
Learning rate (eta) - 0.001, 
L2 regularization (alpha) - 0, 
Epochs - 10, 
Hidden Layer Size - 128, 
No. of layers - 5,
Optimizer - adam, 
Weight Initialization - Xavier, 
Loss Function - Cross Entropy

Best Results -

Validation Accuracy - 0.8822
Test Accuracy - 0.8687
Test error - 0.3771

Link to wandb report - https://wandb.ai/cs22m005/dlasg1/reports/CS6910-Assignment-1-Report--VmlldzozNzQzMzc4
