# Basic Imports
import numpy as np
import pandas as pd
from math import exp
import numpy.random as nr
import os, random, time,copy, re, glob

# Pytorch Imports
import torch, torchvision
from torch import nn, optim
from torch.functional import F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms, datasets

# Tensorflow Imports
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Images and Plt Imports
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display

classes = {0:"Safe driving",1:"Texting(right hand)",2:"Talking on the phone (right hand)", 3:"Texting (left hand)",
           4:"Talking on the phone (left hand)", 5:"Operating the radio", 6:"Drinking", 7:"Reaching behind", 
           8:"Hair and makeup", 9:"Talking to passenger(s)"}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
output1 = '.\\model_ft.pth'
output2= '.\\model.tflite'
__location1__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(output1)))
__location2__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(output2)))
modelWeightsPath= os.path.join(__location1__, 'model_ft.pth')
tflite_model= os.path.join(__location2__, 'model.tflite')

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
test_transforms = transforms.Compose([transforms.Resize(224),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               normalize])

def plot_images(images):

    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize=(20,10))
    for i in range(rows*cols):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.set_title(f'{images[i][1]}')
        ax.imshow(np.array(images[i][0]))
        ax.axis('off')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for (x, y) in iterator:
        
        #variables can already sort of do that by setting the requires_grad=False
        x = Variable(torch.FloatTensor(np.array(x))).to(device)
        y = Variable(torch.LongTensor(y)).to(device)
        
        optimizer.zero_grad() #to clear the gradients from the previous training step
                
        y_pred = model(x)
        
        loss = criterion(y_pred, y)
        
        acc = calculate_accuracy(y_pred, y)
        
        loss.backward()
        
        #makes the optimizer iterate over all parameters (tensors)
        #it is supposed to update and use their internally stored grad to update their values
        optimizer.step() 
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for (x, y) in iterator:

          #variables can already sort of do that by setting the requires_grad=False
            x = Variable(torch.FloatTensor(np.array(x))).to(device)
            y = Variable(torch.LongTensor(y)).to(device)
        
            y_pred = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def fit_model(model, model_name, train_iterator, valid_iterator, optimizer, loss_criterion, epochs):
    """ Fits a dataset to model"""
    best_valid_loss = float('inf')
    
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    
    for epoch in range(epochs):
    
        start_time = time.time()
    
        train_loss, train_acc = train(model, train_iterator, optimizer, loss_criterion, device)
        valid_loss, valid_acc = evaluate(model, valid_iterator, loss_criterion, device)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accs.append(train_acc*100)
        valid_accs.append(valid_acc*100)
    
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'{model_name}.pt')
    
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        
    return pd.DataFrame({f'{model_name}_Training_Loss':train_losses, 
                        f'{model_name}_Training_Acc':train_accs, 
                        f'{model_name}_Validation_Loss':valid_losses, 
                        f'{model_name}_Validation_Acc':valid_accs})

def plot_training_statistics(train_stats, model_name):
    
    fig, axes = plt.subplots(2, figsize=(15,15))
    axes[0].plot(train_stats[f'{model_name}_Training_Loss'], label=f'{model_name}_Training_Loss')
    axes[0].plot(train_stats[f'{model_name}_Validation_Loss'], label=f'{model_name}_Validation_Loss')
    axes[1].plot(train_stats[f'{model_name}_Training_Acc'], label=f'{model_name}_Training_Acc')
    axes[1].plot(train_stats[f'{model_name}_Validation_Acc'], label=f'{model_name}_Validation_Acc')
    
    axes[0].set_xlabel("Number of Epochs"), axes[0].set_ylabel("Loss")
    axes[1].set_xlabel("Number of Epochs"), axes[1].set_ylabel("Accuracy in %")
    
    axes[0].legend(), axes[1].legend()

def imshow(img):
  ''' function to show image '''
  img = img / 2 + 0.5 # unnormalize
  npimg = img.numpy() # convert to numpy objects
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()

def loadModel():
  model_ft = torchvision.models.resnet18(pretrained=True, progress=False)
  num_ftrs = model_ft.fc.in_features          #in_features
  model_ft.fc = torch.nn.Linear(num_ftrs, 10) #nn.Linear(in_features, out_features)
  model_ft.load_state_dict(torch.load(modelWeightsPath,map_location=torch.device(device)))
  model_ft.to(device)
  model_ft.eval()
  return model_ft

model = loadModel()

def PredictBatch(test_data_batch,batch_size):
  test_iterator2 = DataLoader(test_data_batch,shuffle=True,batch_size = batch_size)
  dataiter = iter(test_iterator2)
  images, labels = dataiter.next()
  
  # print images
  imshow(torchvision.utils.make_grid(images))
  print('GroundTruth:')
  print("---"*10)
  print('--\n'.join('%s' % classes[int(labels[j])] for j in range(batch_size)))
  print("***"*20)
  #Prediction
  outputs = model(images.to(device))
  _, predicted = torch.max(outputs, 1)
  print('Predicted:')
  print("---"*10)
  print('--\n'.join('%s' % classes[int(predicted[j])]for j in range(batch_size)))

def PredictClass(imgPath):
  image = Image.open(imgPath)
  x = test_transforms(image)
  x.unsqueeze_(0)
  outputs = model(x.to(device))
  soft_max = torch.nn.Softmax(dim=1)
  probs = soft_max(outputs.data) 
  prob, indices = torch.topk(probs, 1)
  Top_k = indices[0]
  Classes_nameTop_k=[classes[int(Top_k[0])]]
  ProbTop_k=prob[0].tolist()
  ProbTop_k = [round(elem, 5) for elem in ProbTop_k]
  return Classes_nameTop_k[0] , ProbTop_k[0]

class TensorflowLiteClassificationModel:
    def __init__(self, model_path, labels,transform,classes,image_size=224):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self._input_details = self.interpreter.get_input_details()
        self._output_details = self.interpreter.get_output_details()
        self.labels = labels
        self.image_size=image_size
        self.transform= transform
        self.classes=classes
    def run_from_filepath(self, image_path):
        image = Image.open(image_path)
        x = self.transform(image)
        x.unsqueeze_(0)
        x=x.cpu().detach().numpy()
        return self.run(x)

    def run(self, image):
        """
        args:
          image: a (1, image_size, image_size, 3) np.array

        Returns list of [Label, Probability], of type List<str, float>
        """

        self.interpreter.set_tensor(self._input_details[0]["index"], image)
        self.interpreter.invoke()
        tflite_interpreter_output = self.interpreter.get_tensor(self._output_details[0]["index"])
        probabilities = np.array(tflite_interpreter_output[0])
        exp_x=[exp(x) for x in probabilities]
        probabilities=[exp(x)/sum(exp_x) for x in probabilities]
        # create list of ["label", probability], ordered descending probability
        label_to_probabilities = []
        for i, probability in enumerate(probabilities):
            label_to_probabilities.append([self.labels[i], float(probability)])
        pClass=sorted(label_to_probabilities, key=lambda element: element[1])[-1]
        cls= self.classes[pClass[0]]
        p=pClass[1]
        return cls,p


def tfliteModel_Prediction(imgPath):
  
  labels=range(0,10)
  model = TensorflowLiteClassificationModel(tflite_model,labels,test_transforms,classes)
  clss,p= model.run_from_filepath(imgPath)
  
  return clss,p

def tfliteModel_Plot(test_img_dir,nImages=5):
  clss_=[]
  labels=range(0,10)
  model = TensorflowLiteClassificationModel(tflite_model,labels,test_transforms,classes)
  images_list=[os.path.join(test_img_dir, fname) for fname in random.sample(os.listdir(test_img_dir), nImages)]
  for img in images_list:
    clss,p= model.run_from_filepath(img)
    clss_.append((img, clss))
  plotImg(clss_)

  
def plotImg(images):

    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize=(20,10))
    for i in range(rows*cols):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.set_title(f'{images[i][1]}')
        ax.imshow(mpimg.imread(images[i][0]))
        ax.axis('off')

def predMulti_images(test_img_dir,nImages=5):
  clss_=[]
  images_list=[os.path.join(test_img_dir, fname) for fname in random.sample(os.listdir(test_img_dir), nImages)]
  for img in images_list:
    clss,p=PredictClass(img)
    clss_.append((img, clss))
  plotImg(clss_)