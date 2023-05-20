import numpy as np
import math

# Log in to your W&B account
import wandb
import os
os.environ['WAND_NOTEBOOK_NAME']='train'
# !wandb login d327efaa71cd08cf96d51c7e249ccb5eee77cf57

# argparse is used to get inputs from the user on the command line interface
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("-wp",     "--wandb_project",   help="project_name",       type=str,                                                                  default="dlasg1")
parser.add_argument("-we",     "--wandb_entity",    help="entity",             type=str,                                                                  default="cs22m005")
parser.add_argument("-d",      "--dataset",         help="dataset_name",       type=str,   choices=["fashion_mnist","mnist"],                             default="fashion_mnist")
parser.add_argument("-m",      "--momentum",        help="m",                  type=float, choices=[0.5,0.9],                                             default=0.9)
parser.add_argument("-beta",   "--beta",            help="beta",               type=float, choices=[0.5,0.9],                                             default=0.9)
parser.add_argument("-beta1",  "--beta1",           help="beta1",              type=float, choices=[0.5,0.9],                                             default=0.9)
parser.add_argument("-beta2",  "--beta2",           help="beta2",              type=float, choices=[0.999,0.5],                                           default=0.999)
parser.add_argument("-eps",    "--epsilon",         help="epsilon",            type=float, choices=[1e-3,1e-4],                                           default=1e-3)
parser.add_argument("-w_d",    "--weight_decay",    help="alpha",              type=float, choices=[0,0.0005,0.5],                                        default=0)
parser.add_argument("-o",      "--optimizer",       help="loss_function",      type=str,   choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],default="adam")
parser.add_argument("-lr",     "--learning_rate",   help="lr",                 type=float, choices=[1e-4,1e-3],                                           default=1e-3)
parser.add_argument("-e",      "--epochs",          help="epochs",             type=int,   choices=[5,10],                                                default=10)
parser.add_argument("-b",      "--batch_size",      help="batch_size",         type=int,   choices=[1,16,32,64],                                          default=32)
parser.add_argument("-nhl",    "--num_layers",      help="hidden_layer",       type=int,   choices=[3,4,5],                                               default=5)
parser.add_argument("-w_i",    "--weight_init",     help="weight_init",        type=str,   choices=["random","Xavier"],                                   default="Xavier")
parser.add_argument("-a",      "--activation",      help="activation_function",type=str,   choices=["ReLU","tanh","sigmoid"],                             default="ReLU")
parser.add_argument("-sz",     "--hidden_size",     help="hidden_layer_size",  type=int,   choices=[32,64,128],                                           default=128)
parser.add_argument("-l",      "--loss",            help="loss_function",      type=str,   choices=["mean_squared_error", "cross_entropy"],               default="cross_entropy")
args=parser.parse_args()

# setting the values of parameters that are taken from argparse to the variable names used in the code
project_name=args.wandb_project
entity=args.wandb_entity
dataset_name=args.dataset
m=args.momentum
epsilon=args.epsilon
alpha=args.weight_decay
hidden_layer=args.num_layers
activation_function=args.activation
hidden_layer_size=args.hidden_size
loss_function=args.loss
beta=args.beta
beta1=args.beta1
beta2=args.beta2
lr=args.lr
epochs=args.epochs
optimizer=args.optimizer
weight_init=args.weight_init
loss_function=args.loss
batch_size=args.batch_size

# default paramenters used while running a sweep on wandb
# default_parameters=dict(
#     optimizer="adam",
#     lr=1e-3,
#     epochs=10,
#     hidden_layer_size=128,
#     activation_function="ReLU",
#     weight_init="Xavier",
#     hidden_layer=5,
#     batch_size=32,
#     alpha=0
# )
# config=default_parameters
run=wandb.init(project=project_name,entity=entity,name="train",reinit ='True')
# config=wandb.config

classes=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

# reinit_label function is used to convert the label vector into matrix of one_hot_vector of the labels
def reinit_labels(z):
  y=[[0 for i in range(z.max()+1)] for j in range(len(z))]
  for i in range(len(y)):
    y[i][z[i]]=1
  y=np.array(y)
  return y

# importing the training and test dataset 
# if dataset is fashion_mnist
if(dataset_name=="fashion_mnist"):
    from keras.datasets import fashion_mnist
    (X_Train,Y_Train),(X_Test,Y_Test)=fashion_mnist.load_data()
# if dataset is mnist
elif (dataset_name=="mnist"):
    from keras.datasets import mnist
    (X_Train,Y_Train),(X_Test,Y_Test)=mnist.load_data()
else:
  print("invalid input")
  exit()

# reshaing the input data 
dim1=X_Train.shape[0]
dim2=X_Train.shape[1]*X_Train.shape[1]
dim3=X_Test.shape[0]
x_train=X_Train.reshape(dim1,dim2)/255
x_test=X_Test.reshape(dim3,dim2)/255
y_train=reinit_labels(Y_Train)
y_test=reinit_labels(Y_Test)
output_size=y_train.shape[1]

# splitting the input data into training and validation data in the ratio 9:1
val_x = x_train[int(len(x_train)*0.9):,:]
val_y = y_train[int(len(y_train)*0.9):,:]

x_train = x_train[:int(len(x_train)*0.9),:]
y_train = y_train[:int(len(y_train)*0.9),:]

# output_function is softmax
output_function="softmax"


# layer_init contains the model_parameters that we have used for trainig the network
def layer_init(input_size,output_size,hidden_layer_size,hidden_layer,activation_function,output_function):
  n_layers=[]
  # n_layers contains the input and output dimensions at each layer along with the activation function used
  n_layers.append([input_size,hidden_layer_size,activation_function])
  for i in range(hidden_layer-1):
    n_layers.append([hidden_layer_size,hidden_layer_size,activation_function])
  n_layers.append([hidden_layer_size,output_size,output_function])
  return n_layers


# start_weight_and _biaf function contains the initial weights and biases of the network
def start_weights_and_bias(n_layers,weight_init):
#if weight_init method is random
  if weight_init=="random":
    initial_weights=[]
    initial_bias=[]

    for i in range(len(n_layers)):
      w=np.random.uniform(-1,1,(n_layers[i][1],n_layers[i][0]))
      b=np.random.rand(1,n_layers[i][1])
      initial_weights.append(w)
      initial_bias.append(b)

    return initial_weights,initial_bias
  
# if weight_init method is Xavier
  if weight_init=="Xavier":
    initial_weights=[]
    initial_bias=[]

    for i in range(len(n_layers)):
      num=np.sqrt(6/(n_layers[i][1]+n_layers[i][0]))
      w=np.random.uniform(-num,num,(n_layers[i][1],n_layers[i][0]))
      b=np.random.uniform(-num,num,(1,n_layers[i][1]))
      initial_weights.append(w)
      initial_bias.append(b)

    return initial_weights,initial_bias


# activation function provides the output w.r.t. the activation_function that is used in the network
def activation(z,activation_function):
# if input activation function is sigmoid
  if activation_function=="sigmoid":
    return 1.0/(1.0 + np.exp(-z))

# the activation used at output layer is softmax
  if activation_function=="softmax":
    k=[]
    for i in range(z.shape[0]):
      sum=0
    #   normalization by substracting the max value
      idx=np.argmax(z[i])
      z[i]=z[i]-z[i][idx]
      for j in range(z.shape[1]):
          sum+=np.exp(z[i][j])
      k.append(np.exp(z[i])/sum)
    k=np.array(k) 

    return k
  
# if input activation function is tanh
  if activation_function=="tanh":
    return np.tanh(z)
  
# if input activation function is ReLU
  if activation_function=="ReLU":
    return np.maximum(0,z)
  

# activation_derivative provides the derivative of the activation_function that is used for given input
def activation_derivative(z,activation_function):
# if input activation function is sigmoid its derivative is calculated as below
  if activation_function=="sigmoid":
    result=1.0/(1.0+np.exp(-z))
    return result*(1.0-result)
  
# if input activation function is tanh its derivative is calculated as below
  if activation_function=="tanh":
    return 1-(np.tanh(z))**2
  
# if input activation function is ReLU its derivative is calculated as below
  if activation_function=="ReLU":
    relu = np.maximum(0, z)
    relu[relu > 0] = 1
    return relu


# loss function calculates the loss based on the function used
def loss(y_pred,y_batch,loss_function,weight,alpha):
  # epsilon is used to avoid zero inside log
  epsilon=1e-5
# if input loss function is cross_entropy then loss is calculated as below
  if(loss_function=="cross_entropy"):
    loss = -(np.multiply(y_batch, np.log(y_pred+epsilon))).sum() / len(y_pred)
    reg=0
    # calculating the regularization term 
    for i in range(len(weight)):
      reg+=np.sum(weight[i]**2)
    reg=((alpha/2)*reg)/len(y_pred)

    return (loss+reg)
  
# # if input loss function is mean_squared_error then loss is calculated as below
  elif(loss_function=="mean_squared_error"):
    num=np.sum((y_pred-y_batch)**2)
    deno=2*len(y_pred)
    loss=(num/deno)

    return loss


# accuracy function is used to find the accuracy in traing the model
def accuracy(y_pred,y_batch):
  count=0
  for i in range(len(y_pred)):
    id1=np.argmax(y_pred[i])
    id2=np.argmax(y_batch[i])
    # if true label is same as predicted label the increase the accuracy count
    if id1==id2:
      count+=1
  return count


# forward_propogation is used to get the predicted values of the labesl for a given input and the configuration used
def forward_propogation(x_batch,weight,bias,n_layers,activation_function,output_function):
  # a and h are the pre-activation and activation respectively at each layer in the network
  a,h=[],[]
  k=1
#   preactivation a=(W.X)+B
  a_1=np.matmul(x_batch,weight[0].T)+bias[0]
  a.append(a_1)

  for i in range(len(a_1)):
    a_1[i]=a_1[i]/a_1[i][np.argmax(a_1[i])]

# activation h=activation_function(pre-activation)
  h_1=activation(a_1,activation_function)
  h.append(h_1)

# calculating a and h for entire neural network
  for j in range(len(n_layers)-2):
    k+=1
    a.append(np.matmul(h[j],weight[j+1].T)+bias[j+1])
    h.append(activation(a[j+1],activation_function))
  a.append(np.matmul(h[k-1],weight[k].T)+bias[k])
  z=activation(a[k],output_function)
  h.append(z)

  return a,h


# backward_propagation function is used to calculate the derivative w.r.t. weights and biases while movinf from output to input in the network
def backward_propagation(x_batch,y_pred,y_batch,weight,a,h,n_layers,activation_function):
  # dw and db contains derivative w.r.t. weights and biases respectively at each layer in the network
  dw,db={},{}
  m=len(y_batch)
  da_prev=y_pred-y_batch
  i=len(n_layers)-1
  while(i>=1):
    # calculating derivative w.r.t. weights at each layer using the formula taught in class
    da=da_prev
    d=[]
    z=(np.array(np.matmul(h[i-1].T,da)).T)/m
    b=len(z)
    c=len(z[0])
    dw[i+1]=z.reshape(b,c)

    # calculating derivative w.r.t. weights at each layer using the formula taught in class
    for k in range(len(da[0])):
      sum=0;
      for j in range(len(da)):
        sum+=da[j][k]
      d.append(sum/m)
    db[i+1]=np.array(d)
    dh_prev=np.matmul(da,weight[i])
    a_new=(activation_derivative(a[i-1],activation_function))
    da_prev=np.multiply(dh_prev,a_new)
    i-=1
  d=[]
  z=(np.array(np.matmul(x_batch.T,da_prev)).T)/m
  b=len(z)
  c=len(z[0])
  dw[1]=z.reshape(b,c)
  for k in range(len(da_prev[0])):
    sum=0;
    for j in range(len(da_prev)):
      sum+=da_prev[j][k]
    d.append(sum/m)
  db[1]=np.array(d)

  return dw,db


# stochastic_gradient_descent is used to train the model for a batch_size of 1 for a particular network configuration
def stochastic_gradient_descent(x_train,y_train,batches,hidden_layer,hidden_layer_size,lr,weight_init,epochs,activation_function,output_function,loss_function,alpha):
  n_layers=layer_init(dim2,output_size,hidden_layer_size,hidden_layer,activation_function,output_function)

# initializing the weights and biases using the weight_init method
  weight,bias=start_weights_and_bias(n_layers,weight_init)

# splitting the train data into given number of batches
  x_batch = np.array(np.array_split(x_train, batches))
  y_batch = np.array(np.array_split(y_train, batches))
  train_error,train_accuracy,val_error,val_accuracy=[],[],[],[]
  print()
# traing the model for given number of epochs
  for e in range(epochs):
    l,counttrain=0,0

    # training for each batch
    for i in range(len(x_batch)):
      
      # call to forward propagation function to calculate pre-activation and activation at each layer
      a,h=forward_propogation(x_batch[i],weight,bias,n_layers,activation_function,output_function)
      y_pred=h[-1]

      # call to backward_propagation function to calculate derivative w.r.t. weights and biases at each layer
      dw,db=backward_propagation(x_batch[i],y_pred,y_batch[i],weight,a,h,n_layers,activation_function)

      # update rule for stochastic_gradient_descent
      for j in range(len(weight)):
        weight[j]=weight[j]-lr*dw[j+1]
        bias[j]=bias[j]-lr*db[j+1]
    
    # calculating the regularized loss and accuracy on training set
    a,h=forward_propogation(x_train,weight,bias,n_layers,activation_function,output_function)
    y_pred=h[-1]
    l=loss(y_pred,y_train,loss_function,weight,alpha)
    counttrain=accuracy(y_pred,y_train)
    train_error.append(l)
    train_accuracy.append(counttrain/len(x_train))
    print("Train_accuracy at epoch :",e," is--", counttrain/len(x_train))
    print("Train_loss at epoch :",e," is--", l)
    print()

    # calculating the regularized loss and accuracy on validation set
    a,h=forward_propogation(val_x,weight,bias,n_layers,activation_function,output_function)
    y_valpred=h[-1]
    l_val=loss(y_valpred,val_y,loss_function,weight,alpha)
    val_error.append(l_val)
    countval=accuracy(y_valpred,val_y)
    val_accuracy.append(countval/len(val_y))
    print("Validation_accuracy at epoch :",e," is--", countval/len(val_y))
    print("Validation_loss at epoch :",e," is--", l_val)
    print()

    # logging the values obtained in wandb
    wandb.log({"train_accuracy":(counttrain/len(x_train)),"train_error":l,"val_accuracy": (countval/len(val_y)),"val_error":l_val})
  return weight,bias,train_error,train_accuracy,val_error,val_accuracy


# momentum_gradient_descent is used to train the model for a batch_size of 1 for a particular network configuration
def momentum_gd(x_train,y_train,batches,hidden_layer,hidden_layer_size,lr,weight_init,epochs,activation_function,output_function,loss_function,alpha):
  n_layers=layer_init(dim2,output_size,hidden_layer_size,hidden_layer,activation_function,output_function)

  # initializing the weights and biases using the weight_init method
  weight,bias=start_weights_and_bias(n_layers,weight_init)

  # splitting the train data into given number of batches
  x_batch = np.array(np.array_split(x_train, batches))
  y_batch = np.array(np.array_split(y_train, batches))
  print()
  train_error,train_accuracy,val_error,val_accuracy=[],[],[],[]
  history_weight={}
  history_bias={}

  # history_weight and history_bias store the history for each round of updates for weights and biases
  for i in range(len(n_layers)):
    history_weight[i+1]=np.zeros(weight[i].shape)
    history_bias[(i+1)]=np.zeros(bias[i].shape)

  # training the model for given number of epochs
  for e in range(epochs):

    l,counttrain=0,0
    for i in range(len(x_batch)):
      
      # call to forward propagation function to calculate pre-activation and activation at each layer
      a,h=forward_propogation(x_batch[i],weight,bias,n_layers,activation_function,output_function)
      y_pred=h[-1]

      # call to backward_propagation function to calculate derivative w.r.t. weights and biases at each layer
      dw,db=backward_propagation(x_batch[i],y_pred,y_batch[i],weight,a,h,n_layers,activation_function)  

      # update rule for momentum_gd
      for j in range(len(n_layers)):
        history_weight[j+1]=m*history_weight[j+1]+lr*dw[j+1]
        history_bias[j+1]=m*history_bias[j+1]+lr*db[j+1]

        weight[j]=weight[j]-history_weight[j+1]
        bias[j]=bias[j]-history_bias[j+1]

    # calculating the regularized loss and accuracy on training set
    a,h=forward_propogation(x_train,weight,bias,n_layers,activation_function,output_function)
    y_pred=h[-1]
    l=loss(y_pred,y_train,loss_function,weight,alpha)
    counttrain=accuracy(y_pred,y_train)
    train_error.append(l)
    train_accuracy.append(counttrain/len(x_train))
    print("Train_accuracy at epoch :",e," is--", counttrain/len(x_train))
    print("Train_loss at epoch :",e," is--", l)
    print()

    # calculating the regularized loss and accuracy on validation set
    a,h=forward_propogation(val_x,weight,bias,n_layers,activation_function,output_function)
    y_valpred=h[-1]
    l_val=loss(y_valpred,val_y,loss_function,weight,alpha)
    val_error.append(l_val)
    countval=accuracy(y_valpred,val_y)
    val_accuracy.append(countval/len(val_y))
    print("Validation_accuracy at epoch :",e," is--", countval/len(val_y))
    print("Validation_loss at epoch :",e," is--", l_val)
    print()

    # logging the values in wandb
    wandb.log({"train_accuracy":(counttrain/len(x_train)),"train_error":l,"val_accuracy": (countval/len(val_y)),"val_error":l_val})

  return weight,bias,train_error,train_accuracy,val_error,val_accuracy


# nesterov_gradient_descent is used to train the model for a batch_size of 1 for a particular network configuration
def nesterov_gd(x_train,y_train,batches,hidden_layer,hidden_layer_size,lr,weight_init,epochs,activation_function,output_function,loss_function,alpha):
  n_layers=layer_init(dim2,output_size,hidden_layer_size,hidden_layer,activation_function,output_function)

  # initializing the weights and biases using the weight_init method
  weight,bias=start_weights_and_bias(n_layers,weight_init)

  # splitting the train data into given number of batches
  x_batch = np.array(np.array_split(x_train, batches))
  y_batch = np.array(np.array_split(y_train, batches))
  print()
  train_error,train_accuracy,val_error,val_accuracy=[],[],[],[]
  history_weight={}
  history_bias={}

  # history_weight and history_bias store the history for each round of updates for weights and biases
  for i in range(len(n_layers)):
    history_weight[i+1]=np.zeros(weight[i].shape)
    history_bias[i+1]=np.zeros(bias[i].shape)


  # training the model for given number of epochs
  for e in range(epochs):
    l,counttrain=0,0


    for i in range(len(x_batch)):
      
      # lookahead_weights and lookahead_bias contains the weights and biases used in the next round after update
      lookahead_weight=[]
      lookahead_bias=[]
      for j in range(len(n_layers)):
        lookahead_weight.append(weight[j] - m* history_weight[j+1])
        lookahead_bias.append(bias[j] - m* history_bias[j+1])

      # call to forward propagation function to calculate pre-activation and activation at each layer
      a,h=forward_propogation(x_batch[i],lookahead_weight,lookahead_bias,n_layers,activation_function,output_function)
      y_pred=h[-1]

      # call to backward_propagation function to calculate derivative w.r.t. weights and biases at each layer
      dw,db=backward_propagation(x_batch[i],y_pred,y_batch[i],lookahead_weight,a,h,n_layers,activation_function) 

      # update rule for nag 
      for j in range(len(n_layers)):
        history_weight[j+1]=m*history_weight[j+1]+lr*dw[j+1]
        history_bias[j+1]=m*history_bias[j+1]+lr*db[j+1]

        weight[j]=weight[j]-history_weight[j+1]
        bias[j]=bias[j]-history_bias[j+1]

    # calculating the regularized loss and accuracy on training set
    a,h=forward_propogation(x_train,weight,bias,n_layers,activation_function,output_function)
    y_pred=h[-1]
    l=loss(y_pred,y_train,loss_function,weight,alpha)
    counttrain=accuracy(y_pred,y_train)
    train_error.append(l)
    train_accuracy.append(counttrain/len(x_train))
    print("Train_accuracy at epoch :",e," is--", counttrain/len(x_train))
    print("Train_loss at epoch :",e," is--", l)
    print()

    # calculating the regularized loss and accuracy on validation set
    a,h=forward_propogation(val_x,weight,bias,n_layers,activation_function,output_function)
    y_valpred=h[-1]
    l_val=loss(y_valpred,val_y,loss_function,weight,alpha)
    val_error.append(l_val)
    countval=accuracy(y_valpred,val_y)
    val_accuracy.append(countval/len(val_y))
    print("Validation_accuracy at epoch :",e," is--", countval/len(val_y))
    print("Validation_loss at epoch :",e," is--", l_val)
    print()
    # loggin values into wandb
    wandb.log({"train_accuracy":(counttrain/len(x_train)),"train_error":l,"val_accuracy": (countval/len(val_y)),"val_error":l_val})

  return weight,bias,train_error,train_accuracy,val_error,val_accuracy


# rms_prop_gradient_descent is used to train the model for a batch_size of 1 for a particular network configuration
def rms_prop(x_train,y_train,batches,hidden_layer,hidden_layer_size,lr,weight_init,epochs,activation_function,output_function,loss_function,alpha):
  n_layers=layer_init(dim2,output_size,hidden_layer_size,hidden_layer,activation_function,output_function)

  # initializing the weights and biases using the weight_init method
  weight,bias=start_weights_and_bias(n_layers,weight_init)

  # splitting the train data into given number of batches
  x_batch = np.array(np.array_split(x_train, batches))
  y_batch = np.array(np.array_split(y_train, batches))
  print()
  train_error,train_accuracy,val_error,val_accuracy=[],[],[],[]
  history_weight={}
  history_bias={}

  # history_weight and history_bias store the history for each round of updates for weights and biases
  for i in range(len(n_layers)):
    history_weight[i+1]=np.zeros(weight[i].shape)
    history_bias[(i+1)]=np.zeros(bias[i].shape)

  # training the model for given number of epochs
  for e in range(epochs):
    l,counttrain=0,0

    for i in range(len(x_batch)):
      
      # call to forward propagation function to calculate pre-activation and activation at each layer
      a,h=forward_propogation(x_batch[i],weight,bias,n_layers,activation_function,output_function)
      y_pred=h[-1]

      # call to backward_propagation function to calculate derivative w.r.t. weights and biases at each layer
      dw,db=backward_propagation(x_batch[i],y_pred,y_batch[i],weight,a,h,n_layers,activation_function) 

      # update rule for rms_prop_gd
      for j in range(len(n_layers)):
        history_weight[j+1]=beta*history_weight[j+1]+(1-beta)*(dw[j+1])**2
        history_bias[j+1]=beta*history_bias[j+1]+(1-beta)*(db[j+1])**2

        weight[j] -= lr* np.divide(dw[j+1],np.sqrt(history_weight[j+1] + epsilon))
        bias[j] -= lr* np.divide(db[j+1],np.sqrt(history_bias[j+1] + epsilon))

    # calculating the regularized loss and accuracy on training set
    a,h=forward_propogation(x_train,weight,bias,n_layers,activation_function,output_function)
    y_pred=h[-1]
    l=loss(y_pred,y_train,loss_function,weight,alpha)
    counttrain=accuracy(y_pred,y_train)
    train_error.append(l)
    train_accuracy.append(counttrain/len(x_train))
    print("Train_accuracy at epoch :",e," is--", counttrain/len(x_train))
    print("Train_loss at epoch :",e," is--", l)
    print()

    # calculating the regularized loss and accuracy on validation set
    a,h=forward_propogation(val_x,weight,bias,n_layers,activation_function,output_function)
    y_valpred=h[-1]
    l_val=loss(y_valpred,val_y,loss_function,weight,alpha)
    val_error.append(l_val)
    countval=accuracy(y_valpred,val_y)
    val_accuracy.append(countval/len(val_y))
    print("Validation_accuracy at epoch :",e," is--", countval/len(val_y))
    print("Validation_loss at epoch :",e," is--", l_val)
    print()

    # logging values into wandb
    wandb.log({"train_accuracy":(counttrain/len(x_train)),"train_error":l,"val_accuracy": (countval/len(val_y)),"val_error":l_val})

  return weight,bias,train_error,train_accuracy,val_error,val_accuracy


# adam_gradient_descent is used to train the model for a batch_size of 1 for a particular network configuration
def adam(x_train,y_train,batches,hidden_layer,hidden_layer_size,lr,weight_init,epochs,activation_function,output_function,loss_function,alpha):
  n_layers=layer_init(dim2,output_size,hidden_layer_size,hidden_layer,activation_function,output_function)
  
  # initializing the weights and biases using the weight_init method
  weight,bias=start_weights_and_bias(n_layers,weight_init)

  # splitting the train data into given number of batches
  x_batch = np.array(np.array_split(x_train, batches))
  y_batch = np.array(np.array_split(y_train, batches))
  print()
  train_error,train_accuracy,val_error,val_accuracy=[],[],[],[]

  v_weight={}
  v_bias={}
  m_weight={}
  m_bias={}

  v_hatw={}
  v_hatb={}
  m_hatw={}
  m_hatb={}

  # m_weight and m_bias are the momentume terms and v_weight and v_bias are the history terms
  for i in range(len(n_layers)):
    v_weight[i+1]=np.zeros(weight[i].shape)
    v_bias[(i+1)]=np.zeros(bias[i].shape)
    m_weight[i+1]=np.zeros(weight[i].shape)
    m_bias[(i+1)]=np.zeros(bias[i].shape)

  t=0

  # training the model for given number of epochs
  for e in range(epochs):
    l,counttrain=0,0

    for i in range(len(x_batch)):
      t+=1

      # call to forward propagation function to calculate pre-activation and activation at each layer
      a,h=forward_propogation(x_batch[i],weight,bias,n_layers,activation_function,output_function)
      y_pred=h[-1]

      # call to backward_propagation function to calculate derivative w.r.t. weights and biases at each layer
      dw,db=backward_propagation(x_batch[i],y_pred,y_batch[i],weight,a,h,n_layers,activation_function)

      # update rule for adam gd
      for j in range(len(n_layers)):
        v_weight[j+1] = beta2 * v_weight[j+1] + (1-beta2) * (dw[j+1])**2
        v_bias[j+1] = beta2 * v_bias[j+1] + (1-beta2) * (db[j+1])**2

        m_weight[j+1] = beta1 * m_weight[j+1] + (1-beta1) * dw[j+1]
        m_bias[j+1] = beta1 * m_bias[j+1] + (1-beta1) * db[j+1]

        v_hatw[j+1] = np.divide(v_weight[j+1], (1-beta2**t))
        v_hatb[j+1] = np.divide(v_bias[j+1], (1-beta2**t))

        m_hatw[j+1] = np.divide(m_weight[j+1], (1-beta1**t))
        m_hatb[j+1] = np.divide(m_bias[j+1], (1-beta1**t))

        weight[j] -= lr * np.divide(m_hatw[j+1], np.sqrt(v_hatw[j+1] + epsilon))
        bias[j] -= lr * np.divide(m_hatb[j+1], np.sqrt(v_hatb[j+1] + epsilon))
    
    # calculating the regularized loss and accuracy on training set
    a,h=forward_propogation(x_train,weight,bias,n_layers,activation_function,output_function)
    y_pred=h[-1]
    l=loss(y_pred,y_train,loss_function,weight,alpha)
    counttrain=accuracy(y_pred,y_train)
    train_error.append(l)
    train_accuracy.append(counttrain/len(x_train))
    print("Train_accuracy at epoch :",e," is--", counttrain/len(x_train))
    print("Train_loss at epoch :",e," is--", l)
    print()

    # calculating the regularized loss and accuracy on validation set
    a,h=forward_propogation(val_x,weight,bias,n_layers,activation_function,output_function)
    y_valpred=h[-1]
    l_val=loss(y_valpred,val_y,loss_function,weight,alpha)
    val_error.append(l_val)
    countval=accuracy(y_valpred,val_y)
    val_accuracy.append(countval/len(val_y))
    print("Validation_accuracy at epoch :",e," is--", countval/len(val_y))
    print("Validation_loss at epoch :",e," is--", l_val)
    print()

    # logging values into wandb
    wandb.log({"train_accuracy":(counttrain/len(x_train)),"train_error":l,"val_accuracy": (countval/len(val_y)),"val_error":l_val})

  return weight,bias,train_error,train_accuracy,val_error,val_accuracy


# nadam_gradient_descent is used to train the model for a batch_size of 1 for a particular network configuration
def nadam(x_train,y_train,batches,hidden_layer,hidden_layer_size,lr,weight_init,epochs,activation_function,output_function,loss_function,alpha):
  n_layers=layer_init(dim2,output_size,hidden_layer_size,hidden_layer,activation_function,output_function)

  # initializing the weights and biases using the weight_init method
  weight,bias=start_weights_and_bias(n_layers,weight_init)

  # splitting the train data into given number of batches
  x_batch = np.array(np.array_split(x_train, batches))
  y_batch = np.array(np.array_split(y_train, batches))
  print()
  train_error,train_accuracy,val_error,val_accuracy=[],[],[],[]

  v_weight={}
  v_bias={}
  m_weight={}
  m_bias={}

  v_hatw={}
  v_hatb={}
  m_hatw={}
  m_hatb={}

  # m_weight and m_bias are the momentume terms and v_weight and v_bias are the history terms
  for i in range(len(n_layers)):
    v_weight[i+1]=np.zeros(weight[i].shape)
    v_bias[(i+1)]=np.zeros(bias[i].shape)
    m_weight[i+1]=np.zeros(weight[i].shape)
    m_bias[(i+1)]=np.zeros(bias[i].shape)

  t=0

  # training the model for given number of epochs
  for e in range(epochs):
    l,counttrain=0,0

    # lookahead terms contains the weights and biases and momentum and history used in the next round after update
    for i in range(len(x_batch)):
      
      lookahead_w=[]
      lookahead_b=[]
      lookahead_mhatw=[]
      lookahead_mhatb=[]
      lookahead_vhatw=[]
      lookahead_vhatb=[]
      t+=1

      for j in range(len(n_layers)):
        lookahead_vhatw.append(np.divide(beta2 * v_weight[j+1], (1 - beta2 ** t)))
        lookahead_vhatb.append(np.divide(beta2 * v_bias[j+1], (1 - beta2 ** t)))

        lookahead_mhatw.append(np.divide(beta1 * m_weight[j+1], (1 - beta1 ** t)))
        lookahead_mhatb.append(np.divide(beta1 * m_bias[j+1], (1 - beta1 ** t)))

        lookahead_w.append(weight[j] - lr*np.divide(lookahead_mhatw[j], np.sqrt(lookahead_vhatw[j] + epsilon)))
        lookahead_b.append(bias[j] - lr*np.divide(lookahead_mhatb[j], np.sqrt(lookahead_vhatb[j] + epsilon)))
      
      # call to forward propagation function to calculate pre-activation and activation at each layer
      a,h=forward_propogation(x_batch[i],lookahead_w,lookahead_b,n_layers,activation_function,output_function)
      y_pred=h[-1]

      # call to backward_propagation function to calculate derivative w.r.t. weights and biases at each layer
      dw,db=backward_propagation(x_batch[i],y_pred,y_batch[i],lookahead_w,a,h,n_layers,activation_function)  
      
      # update rule for nadam gd
      for j in range(len(n_layers)):
        v_weight[j+1] = beta2 * v_weight[j+1] + (1-beta2) * (dw[j+1])**2
        v_bias[j+1] = beta2 * v_bias[j+1] + (1-beta2) * (db[j+1])**2

        m_weight[j+1] = beta1 * m_weight[j+1] + (1-beta1) * dw[j+1]
        m_bias[j+1] = beta1 * m_bias[j+1] + (1-beta1) * db[j+1]

        v_hatw[j+1] = np.divide(v_weight[j+1], (1-beta2**t))
        v_hatb[j+1] = np.divide(v_bias[j+1], (1-beta2**t))

        m_hatw[j+1] = np.divide(m_weight[j+1], (1-beta1**t))
        m_hatb[j+1] = np.divide(m_bias[j+1], (1-beta1**t))

        weight[j] -= lr * np.divide(m_hatw[j+1], np.sqrt(v_hatw[j+1] + epsilon))
        bias[j] -= lr * np.divide(m_hatb[j+1], np.sqrt(v_hatb[j+1] + epsilon))

    # calculating the regularized loss and accuracy on training set
    a,h=forward_propogation(x_train,weight,bias,n_layers,activation_function,output_function)
    y_pred=h[-1]
    l=loss(y_pred,y_train,loss_function,weight,alpha)
    counttrain=accuracy(y_pred,y_train)
    train_error.append(l)
    train_accuracy.append(counttrain/len(x_train))
    print("Train_accuracy at epoch :",e," is--", counttrain/len(x_train))
    print("Train_loss at epoch :",e," is--", l)
    print()

    # calculating the regularized loss and accuracy on validation set
    a,h=forward_propogation(val_x,weight,bias,n_layers,activation_function,output_function)
    y_valpred=h[-1]
    l_val=loss(y_valpred,val_y,loss_function,weight,alpha)
    val_error.append(l_val)
    countval=accuracy(y_valpred,val_y)
    val_accuracy.append(countval/len(val_y))
    print("Validation_accuracy at epoch :",e," is--", countval/len(val_y))
    print("Validation_loss at epoch :",e," is--", l_val)
    print()

    # logging values into wandb
    wandb.log({"train_accuracy":(counttrain/len(x_train)),"train_error":l,"val_accuracy": (countval/len(val_y)),"val_error":l_val})

  return weight,bias,train_error,train_accuracy,val_error,val_accuracy


# the train function contains calls to various optimization function based on the input given
def train(x_train,y_train,batch_size,hidden_layer,hidden_layer_size,lr,weight_init,epochs,activation_function,output_function,optimizer,loss_function,alpha):
  
  batches=math.ceil(len(x_train)/batch_size)
#   print("batch_size :",batch_size)

  # if optimizer=sgd then stochastic gradient descent will be executed
  if optimizer=="sgd":
    # print("stochastic_gd")
    weight,bias,train_error,train_accuracy,val_error,val_accuracy=stochastic_gradient_descent(x_train,y_train,batches,hidden_layer,hidden_layer_size,lr,weight_init,epochs,activation_function,output_function,loss_function,alpha)
  
  # if optimizer=momentum then momentum gradient descent will be executed
  elif optimizer=="momentum":
    # print("momentum_gd")
    weight,bias,train_error,train_accuracy,val_error,val_accuracy=momentum_gd(x_train,y_train,batches,hidden_layer,hidden_layer_size,lr,weight_init,epochs,activation_function,output_function,loss_function,alpha)
  
  # if optimizer=nesterov then nesterov gradient descent will be executed
  elif optimizer=="nag":
    # print("nesterov_gd")
    weight,bias,train_error,train_accuracy,val_error,val_accuracy=nesterov_gd(x_train,y_train,batches,hidden_layer,hidden_layer_size,lr,weight_init,epochs,activation_function,output_function,loss_function,alpha)
  
  # if optimizer=rmsprop then rms_prop gradient descent will be executed
  elif optimizer=="rmsprop":
    # print("rms_prop")
    weight,bias,train_error,train_accuracy,val_error,val_accuracy=rms_prop(x_train,y_train,batches,hidden_layer,hidden_layer_size,lr,weight_init,epochs,activation_function,output_function,loss_function,alpha)
  
  # if optimizer=adam then adam gradient descent will be executed
  elif optimizer=="adam":
    # print("adam")
    weight,bias,train_error,train_accuracy,val_error,val_accuracy=adam(x_train,y_train,batches,hidden_layer,hidden_layer_size,lr,weight_init,epochs,activation_function,output_function,loss_function,alpha)
  
  # if optimizer=nadam then nadam gradient descent will be executed
  elif optimizer=="nadam":
    # print("nadam")
    weight,bias,train_error,train_accuracy,val_error,val_accuracy=nadam(x_train,y_train,batches,hidden_layer,hidden_layer_size,lr,weight_init,epochs,activation_function,output_function,loss_function,alpha)
    
    return weight,bias,train_error,train_accuracy,val_error,val_accuracy

# wandb config used to execute a sweep in wandb
# lr = config.lr
# epochs = config.epochs
# batch_size = config.batch_size
# optimizer = config.optimizer
# weight_init = config.weight_init
# hidden_layer_size=config.hidden_layer_size
# activation_function=config.activation_function
# hidden_layer=config.hidden_layer
# alpha=config.alpha
# run.name='hl_'+str(hidden_layer)+'_bs_'+str(batch_size)+'_ac_'+activation_function

# call to train function
weight,bias,train_error,train_accuracy,val_error,val_accuracy=train(x_train,y_train,batch_size,hidden_layer,hidden_layer_size,lr,weight_init,epochs,activation_function,output_function,optimizer,loss_function,alpha)


n_layers=layer_init(dim2,output_size,hidden_layer_size,hidden_layer,activation_function,output_function)
for i in range(len(n_layers)):
    print(n_layers[i])
a,h=forward_propogation(x_test,weight,bias,n_layers,activation_function,output_function)
y_pred=h[-1]
y=[]
for i in range(len(y_pred)):
  y.append(np.argmax(y_pred[i]))
l_test=loss(y_pred,y_test,loss_function,weight,alpha)
counttest=accuracy(y_pred,y_test)

print("Test_accuracy after training model is--", counttest/len(y_test))
print("Test_loss after training model is--", l_test)

# the following the the code for making the confusion matrix on test data as shown in the report

# cm=wandb.plot.confusion_matrix(
#   y_true=Y_Test,
#   preds=y,
#   class_names= classes
# )
# print('Test Confusion Matrix\n')
# wandb.log({"conf_mat": cm})

