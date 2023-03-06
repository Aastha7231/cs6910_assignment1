import numpy as np
# Log in to your W&B account
import wandb
import os
from keras.datasets import fashion_mnist
os.environ['WAND_NOTEBOOK_NAME']='train'

# !wandb login d327efaa71cd08cf96d51c7e249ccb5eee77cf57
wandb.init(project='deeplearning',entity='cs22m005',name='train')
classes=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']



import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--optimizer",help="loss_function",type=str,choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
parser.add_argument("--lr",help="lr",type=float, choices=[1e-2,1e-3])
parser.add_argument("--epochs",help="epochs",type=int,choices=[5,10])
parser.add_argument("--batch_size",help="batch_size",type=int,choices=[1,16,32,64])
parser.add_argument("--num_layers",help="hidden_layer",type=int,choices=[3,4,5])
parser.add_argument("--weight_init",help="weight_init",type=str,choices=["random","Xavier"])
parser.add_argument("--activation",help="activation_function",type=str,choices=["reLU","tanh","sigmoid"])
parser.add_argument("--hidden_size",help="hidden_layer_size",type=int,choices=[32,64,128])
parser.add_argument("--loss",help="loss_function",type=str,choices=["mean_squared_error", "cross_entropy"])
args=parser.parse_args()

optimizer=args.optimizer
if(args.lr==None):
  lr=1e-2
else: lr=(args.lr)
if(args.epochs==None):
  epochs=5
else: epochs=args.epochs
if (args.batch_size==None):
  batch_size=32
else: batch_size=args.batch_size
if (args.num_layers==None):
  hidden_layer=3
else: hidden_layer=args.num_layers
if (args.weight_init==None):
  weight_init="random"
else: weight_init=args.weight_init
if (args.activation==None):
  activation_function="tanh"
else: activation_function=args.activation
if(args.hidden_size==None):
  hidden_layer_size=3
else: hidden_layer_size=int(args.hidden_size)
if(args.loss==None):
  loss_function="cross_entropy"
else: loss_function=args.loss
output_function="softmax"


def reinit_labels(z):
  y=[[0 for i in range(z.max()+1)] for j in range(len(z))]
  for i in range(len(y)):
    y[i][z[i]]=1
  y=np.array(y)
  return y


(X_Train,Y_Train),(X_Test,Y_Test)=fashion_mnist.load_data()
dim1=X_Train.shape[0]
dim2=X_Train.shape[1]*X_Train.shape[1]
dim3=X_Test.shape[0]
x_train=X_Train.reshape(dim1,dim2)
x_test=X_Test.reshape(dim3,dim2)
y_train=reinit_labels(Y_Train)
y_test=reinit_labels(Y_Test)
output_size=y_train.shape[1]


def layer_init(input_size,output_size,hidden_layer_size,hidden_layer,activation_function,output_function):
  n_layers=[]
  n_layers.append([input_size,hidden_layer_size,activation_function])
  for i in range(hidden_layer-1):
    n_layers.append([hidden_layer_size,hidden_layer_size,activation_function])
  n_layers.append([hidden_layer_size,output_size,output_function])
  return n_layers


def start_weights_and_bias(n_layers):
  initial_weights=[]
  initial_bias=[]
  for i in range(len(n_layers)):
    w=np.random.uniform(-1,1,(n_layers[i][1],n_layers[i][0]))
    b=np.random.rand(1,n_layers[i][1])
    initial_weights.append(w)
    initial_bias.append(b)
  return initial_weights,initial_bias


def activation(z,activation_function):
  if activation_function=="sigmoid":
    return 1.0/(1.0 + np.exp(-z))
  if activation_function=="softmax":
    k=[]
    for i in range(z.shape[0]):
      sum=0
      for j in range(z.shape[1]):
          sum+=np.exp(z[i][j])
      k.append(np.exp(z[i])/sum)
    k=np.array(k) 
    return k
  if activation_function=="tanh":
    return np.tanh(z)


def activation_derivative(z,activation_function):
  if activation_function=="sigmoid":
    result=1.0/(1.0+np.exp(-z))
    return result*(1.0-result)
  if activation_function=="tanh":
    return 1-(np.tanh(z))**2


def train_loss(y_pred,y_batch):
  loss = -(np.multiply(y_batch, np.log(y_pred))).sum() / len(y_pred)
  return loss


def train_accuracy(y_pred,y_batch):
  count=0
  for i in range(len(y_pred)):
    id1=np.argmax(y_pred[i])
    id2=np.argmax(y_batch[i])
    if id1==id2:
      count+=1
  return count


def forward_propogation(x_batch,y_train,weight,bias,n_layers,activation_function,output_function):
  a,h=[],[]
  k=1
  a_1=np.matmul(x_batch,weight[0].T)+bias[0]
  a.append(a_1)
  for i in range(len(a_1)):
    a_1[i]=a_1[i]/a_1[i][np.argmax(a_1[i])]
  h_1=activation(a_1,activation_function)
  h.append(h_1)
  for j in range(len(n_layers)-2):
    k+=1
    a.append(np.matmul(h[j],weight[j+1].T)+bias[j+1])
    h.append(activation(a[j+1],activation_function))
  a.append(np.matmul(h[k-1],weight[k].T)+bias[k])
  z=activation(a[k],output_function)
  h.append(z)
  return a,h


def backward_propagation(x_batch,y_pred,y_batch,weight,bias,a,h,n_layers,activation_function):
  dw,db={},{}
  m=len(y_batch)
  da_prev=y_pred-y_batch
  i=len(n_layers)-1
  while(i>=1):
    da=da_prev
    d=[]
    z=(np.array(np.matmul(h[i-1].T,da)).T)/m
    b=len(z)
    c=len(z[0])
    dw[i+1]=z.reshape(b,c)
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


def gradient_descent(x_train,y_train,x_test,y_test,batches,hidden_layer=3,hidden_layer_size=128,lr=0.1,weight_init="random",epochs=1,activation_function="sigmoid",output_function="softmax"):
  n_layers=layer_init(dim2,output_size,hidden_layer_size,hidden_layer,activation_function,output_function)
  for i in range(len(n_layers)):
    print(n_layers[i])
  weight,bias=start_weights_and_bias(n_layers)
  x_batch = np.array(np.array_split(x_train, batches))
  y_batch = np.array(np.array_split(y_train, batches))
  y_hat=[]
  count=0
  for e in range(epochs):
    loss=0
    for i in range(len(x_batch)):
      a,h=forward_propogation(x_batch[i],y_train,weight,bias,n_layers,activation_function,output_function)
      y_pred=h[-1]
      dw,db=backward_propagation(x_batch[i],y_pred,y_batch[i],weight,bias,a,h,n_layers,activation_function)
      for j in range(len(weight)):
        weight[j]=weight[j]-lr*dw[j+1]
        bias[j]=bias[j]-lr*db[j+1]
      loss+=train_loss(y_pred,y_batch[i])
      if(e==epochs-1):
        count+=train_accuracy(y_pred,y_batch[i])
    loss=loss/batches
    print(e,"--> ",loss)
  print("train accuracy : ",count/len(x_train)) 
  a,h=forward_propogation(x_test,y_test,weight,bias,n_layers,activation_function,output_function)
  y_pred=h[-1]
  count=train_accuracy(y_pred,y_test)
  print("test accuracy : ",count/len(y_test))
  
def momentum_gd(x_train,y_train,x_test,y_test,batches,hidden_layer=3,hidden_layer_size=128,lr=0.1,weight_init="random",epochs=1,activation_function="sigmoid",output_function="softmax"):
  n_layers=layer_init(dim2,output_size,hidden_layer_size,hidden_layer,activation_function,output_function)
  # for i in range(len(n_layers)):
  #   print(n_layers[i])
  weight,bias=start_weights_and_bias(n_layers)
  x_batch = np.array(np.array_split(x_train, batches))
  y_batch = np.array(np.array_split(y_train, batches))
  history_weight={}
  history_bias={}
  beta=0.9

  for i in range(len(n_layers)):
    history_weight[i+1]=np.zeros(weight[i].shape)
    history_bias[(i+1)]=np.zeros(bias[i].shape)
  count=0
  for e in range(epochs):
    loss=0
    for i in range(len(x_batch)):
      a,h=forward_propogation(x_batch[i],y_train,weight,bias,n_layers,activation_function,output_function)
      y_pred=h[-1]
      dw,db=backward_propagation(x_batch[i],y_pred,y_batch[i],weight,bias,a,h,n_layers,activation_function)  
      for j in range(len(n_layers)):
        history_weight[j+1]=beta*history_weight[j+1]+lr*dw[j+1]
        history_bias[j+1]=beta*history_bias[j+1]+lr*db[j+1]

        weight[j]=weight[j]-history_weight[j+1]
        bias[j]=bias[j]-history_bias[j+1]
      loss+=train_loss(y_pred,y_batch[i])
      if(e==epochs-1):
        count+=train_accuracy(y_pred,y_batch[i])
    loss=loss/batches
    print(e,"--> ",loss)
  print("train_accuracy : ",count/len(x_train))
  a,h=forward_propogation(x_test,y_test,weight,bias,n_layers,activation_function,output_function)
  y_pred=h[-1]
  count=train_accuracy(y_pred,y_test)
  print("test_accuracy :",count/len(y_test))
 
def nesterov_gd(x_train,y_train,x_test,y_test,batches,hidden_layer=3,hidden_layer_size=128,lr=0.1,weight_init="random",epochs=1,activation_function="sigmoid",output_function="softmax"):
  n_layers=layer_init(dim2,output_size,hidden_layer_size,hidden_layer,activation_function,output_function)
  # for i in range(len(n_layers)):
  #   print(n_layers[i])
  weight,bias=start_weights_and_bias(n_layers)
  x_batch = np.array(np.array_split(x_train, batches))
  y_batch = np.array(np.array_split(y_train, batches))
  history_weight={}
  history_bias={}
  beta=0.9

  for i in range(len(n_layers)):
    history_weight[i+1]=np.zeros(weight[i].shape)
    history_bias[i+1]=np.zeros(bias[i].shape)
  count=0
  for e in range(epochs):
    loss=0
    for i in range(len(x_batch)):
      lookahead_weight=[]
      lookahead_bias=[]
      for j in range(len(n_layers)):
        lookahead_weight.append(weight[j] - beta* history_weight[j+1])
        lookahead_bias.append(bias[j] - beta* history_bias[j+1])
      a,h=forward_propogation(x_batch[i],y_train,weight,bias,n_layers,activation_function,output_function)
      y_pred=h[-1]
      dw,db=backward_propagation(x_batch[i],y_pred,y_batch[i],weight,bias,a,h,n_layers,activation_function)  
      for j in range(len(n_layers)):
        history_weight[j+1]=beta*history_weight[j+1]+lr*dw[j+1]
        history_bias[j+1]=beta*history_bias[j+1]+lr*db[j+1]

        weight[j]=weight[j]-history_weight[j+1]
        bias[j]=bias[j]-history_bias[j+1]
      loss+=train_loss(y_pred,y_batch[i])
      if(e==epochs-1):
        count+=train_accuracy(y_pred,y_batch[i])
    loss=loss/batches
    print(e,"--> ",loss)
  print("train_accuracy :",count/len(x_train))
  a,h=forward_propogation(x_test,y_test,weight,bias,n_layers,activation_function,output_function)
  y_pred=h[-1]
  count=train_accuracy(y_pred,y_test)
  print("test_accuracy:",count/len(y_test))

 def rms_prop(x_train,y_train,x_test,y_test,batches,hidden_layer=3,hidden_layer_size=128,lr=0.1,weight_init="random",epochs=1,activation_function="sigmoid",output_function="softmax"):
  n_layers=layer_init(dim2,output_size,hidden_layer_size,hidden_layer,activation_function,output_function)
  # for i in range(len(n_layers)):
  #   print(n_layers[i])
  weight,bias=start_weights_and_bias(n_layers)
  x_batch = np.array(np.array_split(x_train, batches))
  y_batch = np.array(np.array_split(y_train, batches))
  history_weight={}
  history_bias={}
  beta=0.9
  epsilon=1e-3
  for i in range(len(n_layers)):
    history_weight[i+1]=np.zeros(weight[i].shape)
    history_bias[(i+1)]=np.zeros(bias[i].shape)
  count=0
  for e in range(epochs):
    loss=0
    for i in range(len(x_batch)):
      a,h=forward_propogation(x_batch[i],y_train,weight,bias,n_layers,activation_function,output_function)
      y_pred=h[-1]
      dw,db=backward_propagation(x_batch[i],y_pred,y_batch[i],weight,bias,a,h,n_layers,activation_function)  
      for j in range(len(n_layers)):
        history_weight[j+1]=beta*history_weight[j+1]+(1-beta)*(dw[j+1])**2
        history_bias[j+1]=beta*history_bias[j+1]+(1-beta)*(db[j+1])**2

        weight[j] -= lr* np.divide(dw[j+1],np.sqrt(history_weight[j+1] + epsilon))
        bias[j] -= lr* np.divide(db[j+1],np.sqrt(history_bias[j+1] + epsilon))
      loss+=train_loss(y_pred,y_batch[i])
      if(e==epochs-1):
        count+=train_accuracy(y_pred,y_batch[i])
    loss=loss/batches
    print(e,"--> ",loss)
  print("train_accuracy :",count/len(x_train))
  a,h=forward_propogation(x_test,y_test,weight,bias,n_layers,activation_function,output_function)
  y_pred=h[-1]
  count=train_accuracy(y_pred,y_test)
  print("test_accuracy :",count/len(y_test))
  
 def adam(x_train,y_train,x_test,y_test,batches,hidden_layer=3,hidden_layer_size=128,lr=0.1,weight_init="random",epochs=1,activation_function="sigmoid",output_function="softmax"):
  n_layers=layer_init(dim2,output_size,hidden_layer_size,hidden_layer,activation_function,output_function)
  # for i in range(len(n_layers)):
  #   print(n_layers[i])
  weight,bias=start_weights_and_bias(n_layers)
  x_batch = np.array(np.array_split(x_train, batches))
  y_batch = np.array(np.array_split(y_train, batches))

  v_weight={}
  v_bias={}
  m_weight={}
  m_bias={}

  v_hatw={}
  v_hatb={}
  m_hatw={}
  m_hatb={}

  beta1=0.9
  beta2=0.999
  epsilon=1e-3
  for i in range(len(n_layers)):
    v_weight[i+1]=np.zeros(weight[i].shape)
    v_bias[(i+1)]=np.zeros(bias[i].shape)
    m_weight[i+1]=np.zeros(weight[i].shape)
    m_bias[(i+1)]=np.zeros(bias[i].shape)


  count,t=0,0
  for e in range(epochs):
    loss=0
    for i in range(len(x_batch)):
      t+=1
      a,h=forward_propogation(x_batch[i],y_train,weight,bias,n_layers,activation_function,output_function)
      y_pred=h[-1]
      dw,db=backward_propagation(x_batch[i],y_pred,y_batch[i],weight,bias,a,h,n_layers,activation_function)  

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

      loss+=train_loss(y_pred,y_batch[i])
      if(e==epochs-1):
        count+=train_accuracy(y_pred,y_batch[i])
    loss=loss/batches
    print(e,"--> ",loss)
  print("train_accuracy :",count/len(x_train))
  a,h=forward_propogation(x_test,y_test,weight,bias,n_layers,activation_function,output_function)
  y_pred=h[-1]
  count=train_accuracy(y_pred,y_test)
  print("test_accuracy :",count/len(y_test))
  
 def nadam(x_train,y_train,x_test,y_test,batches,hidden_layer=3,hidden_layer_size=128,lr=0.1,weight_init="random",epochs=1,activation_function="sigmoid",output_function="softmax"):
  n_layers=layer_init(dim2,output_size,hidden_layer_size,hidden_layer,activation_function,output_function)
  # for i in range(len(n_layers)):
  #   print(n_layers[i])
  weight,bias=start_weights_and_bias(n_layers)
  x_batch = np.array(np.array_split(x_train, batches))
  y_batch = np.array(np.array_split(y_train, batches))

  v_weight={}
  v_bias={}
  m_weight={}
  m_bias={}

  v_hatw={}
  v_hatb={}
  m_hatw={}
  m_hatb={}

  beta1=0.9
  beta2=0.999
  epsilon=1e-3
  for i in range(len(n_layers)):
    v_weight[i+1]=np.zeros(weight[i].shape)
    v_bias[(i+1)]=np.zeros(bias[i].shape)
    m_weight[i+1]=np.zeros(weight[i].shape)
    m_bias[(i+1)]=np.zeros(bias[i].shape)


  count,t=0,0
  for e in range(epochs):
    loss=0
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
      
      a,h=forward_propogation(x_batch[i],y_train,lookahead_w,lookahead_b,n_layers,activation_function,output_function)
      y_pred=h[-1]
      dw,db=backward_propagation(x_batch[i],y_pred,y_batch[i],lookahead_w,lookahead_b,a,h,n_layers,activation_function)  

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

      loss+=train_loss(y_pred,y_batch[i])
      if(e==epochs-1):
        count+=train_accuracy(y_pred,y_batch[i])
    loss=loss/batches
    print(e,"--> ",loss)
  print("train_accuracy :",count/len(x_train))
  a,h=forward_propogation(x_test,y_test,weight,bias,n_layers,activation_function,output_function)
  y_pred=h[-1]
  count=train_accuracy(y_pred,y_test)
  print("test_accuracy :",count/len(y_test))
  
  
  
  
def train(x_train,y_train,x_test,y_test,batch_size=32,hidden_layer=3,hidden_layer_size=128,lr=0.1,weight_init="random",epochs=1,activation_function="sigmoid",output_function="softmax"):
  batches=len(x_train)/batch_size
  print("batch_gd")
  gradient_descent(x_train,y_train,x_test,y_test,batches,3,128,0.001,"random",10,"tanh","softmax")
  
  print("momentum_gd")
  momentum_gd(x_train,y_train,x_test,y_test,batches,3,128,0.1,"random",10,"sigmoid","softmax")
  
  print("nesterov_gd")
  nesterov_gd(x_train,y_train,x_test,y_test,batches,3,128,0.001,"random",5,"sigmoid","softmax")
  
  print("rms_prop")
  rms_prop(x_train,y_train,x_test,y_test,batches,3,128,0.001,"random",5,"sigmoid","softmax")
  
  print("adam")
  adam(x_train,y_train,x_test,y_test,batches,3,128,0.01,"random",10,"tanh","softmax")
  
  print("nadam")
  nadam(x_train,y_train,x_test,y_test,batches,3,128,0.01,"random",10,"tanh","softmax")

train(x_train,y_train,x_test,y_test,32,3,128,0.1,"random",1,"sigmoid","softmax")
