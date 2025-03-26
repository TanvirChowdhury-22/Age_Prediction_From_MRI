import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torchvision.transforms.functional as TF
from numpy.random import default_rng
#################
#from model import *
import subprocess
from feauture_extractor import *
################
# Define constants
image_height = 128
image_width = 128
num_classes = 1  # For regression problem
num_epochs = 1000
batch_size = 16

# Define the CNN model

# Instantiate the model
#model = CNN()
#model_fx           = Resnet50WithFPN()
model_fc           = torch.load('model_age_prediction.pt') #FNN()
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model_fc.parameters(),lr=0.001)

# Load and preprocess dataset
data_dir = 'Data_brain'
csv_file = 'gas csv.csv'

target_size = (image_height, image_width)
dataset     = CustomDataset(data_dir, csv_file, target_size)
train_size  = int(0.95 * len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Visualize the dataset
#visualize_dataset(dataset)
save_model_path = 'save_model_weights/model_age_prediction.pt'
prev_loss_value =  0 
loss_values =   []
# Training loop



#input           = np.random.randint(low=0, high=255, size = (120,128, 128,1)).astype(np.uint8)
#output_features = model(input)
##################################
for epoch in range(num_epochs):
    #model_fc.train()
    running_loss = 0.0
    for images, labels in train_loader:
        features      =   images.float()  # Add channel dimension
        #features    =   model_fx(images)
        outputs     =   model_fc(images)[:,0]    
        labels      =   labels.float()
        
        optimizer.zero_grad()
        
        loss        = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    # Print average loss per epoch
    torch.save(model_fc, 'model_age_prediction.pt')
    results_val             = subprocess.check_output(f'python3 testing_fc.py',shell=True)
    loss_values.append(running_loss / len(train_loader.dataset))
    
    if prev_loss_value==0: prev_loss_value  =   float(results_val.split()[4])
    elif prev_loss_value<float(results_val.split()[4]):
        print("best r2 = ", float(results_val.split()[4]))
        torch.save(model_fc, save_model_path)
        prev_loss_value     =   float(results_val.split()[4])

    print(f"Epoch {epoch+1}, Training Loss: {loss_values[-1]}")

plt.plot([epoch for epoch in range(num_epochs)], loss_values, 'ro-') 
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.xlabel("Loss")
plt.grid()
plt.savefig('Training_Loss.png')
plt.show()
# Validation loop
model_fc.eval()
running_loss = 0.0
loss_values_val = []
predicted_brain_age = np.array([])
chronological_age = np.array([])
with torch.no_grad():
    for images, labels in val_loader:
        images = images.unsqueeze(1).float()  # Add channel dimension
        
        labels = labels.float().unsqueeze(1)
        try: chronological_age = np.append(chronological_age,labels.detach().numpy())
        except: chronological_age = labels.detach().numpy()

        outputs = model_fc(images)
        try: predicted_brain_age = np.append(predicted_brain_age,outputs.detach().numpy())
        except: predicted_brain_age = outputs.detach().numpy()
        
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        loss_values_val.append(running_loss)
# Print validation loss
plt.figure()
plt.plot([loss for loss in range(len(loss_values_val))], np.array(loss_values_val)/len(val_loader.dataset), 'g*')
plt.title("Validation Loss")
plt.xlabel("Samples")
plt.xlabel("Loss")
plt.savefig('Validation_Loss.png')
plt.grid()
plt.show()

plt.figure()
plt.plot(np.sort(chronological_age),predicted_brain_age[np.argsort(chronological_age)],'b o')
plt.title("Actual vs Predicted Brain Age")
plt.xlabel("Chronological age (weeks)")
plt.ylabel("Predicted Brain age(weeks)")
plt.savefig('actual_vs_predicted_Loss.png')
plt.grid()
plt.show()

print(f"Validation Loss: {running_loss / len(val_loader.dataset)}")
