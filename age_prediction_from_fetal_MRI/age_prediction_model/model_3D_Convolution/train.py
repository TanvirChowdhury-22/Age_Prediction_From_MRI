import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
#################
from model import *
import subprocess
################
# Define constants
image_height = 128
image_width = 128
num_classes = 1  # For regression problem
num_epochs = 800
batch_size = 16

# Define the CNN model

# Instantiate the model
model = CNN()
#model = torch.load('model_age_prediction.pt')
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.0001)

# Load and preprocess dataset
data_dir = 'data/training'
csv_file = 'data/gas csv.csv'

target_size = (image_height, image_width)
dataset = CustomDataset(data_dir, csv_file, target_size)
train_size = int(0.95 * len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Visualize the dataset
#visualize_dataset(dataset)
save_model_path =  'save_model_weights/model_age_prediction.pt'
prev_loss_value =  0 
loss_values     =  []
# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.unsqueeze(1).float()  # Add channel dimension
        labels = labels.float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    # Print average loss per epoch
    loss_values.append(running_loss / len(train_loader.dataset))
    torch.save(model, 'model_age_prediction.pt')
    results_val             = subprocess.check_output(f'python3 testing.py',shell=True)
    
    if prev_loss_value==0: prev_loss_value  =   float(results_val.split()[4])
    elif prev_loss_value<float(results_val.split()[4]):
        print("best r2 = ", float(results_val.split()[4]))
        torch.save(model, save_model_path)
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
model.eval()
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

        outputs = model(images)
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
