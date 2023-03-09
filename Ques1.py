import wandb
import os
from keras.datasets import fashion_mnist

os.environ['WAND_NOTEBOOK_NAME']='ques1'
!wandb login d327efaa71cd08cf96d51c7e249ccb5eee77cf57

(X_train,Y_train),(X_test,Y_test)=fashion_mnist.load_data()

wandb.init(project='dlassignment1',entity='cs22m005',name='ques1')
classes=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
sample_image=[]
label=[]
for i in range(len(X_train)):
  if(len(label)==10):
    break
  if(classes[Y_train[i]] in label):
    continue
  else:
    sample_image.append(X_train[i])
    label.append(classes[Y_train[i]])

wandb.log({"Question 1-Sample Images": [wandb.Image(img, caption=lbl) for img,lbl in zip(sample_image,label)]})
