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
import math
# Define constants


from torchvision import transforms

# Define your transformations for data augmentation
def aug_data(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy array to PIL Image
        transforms.RandomResizedCrop(size=(128, 128),scale=(0.975,0.975)),  # randomly resize      
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomVerticalFlip(),    # Randomly flip the image vertically
        transforms.RandomRotation(90),   # Randomly rotate the image by up to 90 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Randomly jitter brightness and contrast
        transforms.ToTensor()           # Convert PIL Image back to tensor
    ])

    return transform(image)

class CNN(nn.Module):
    def __init__(self, factor=2, base_filters=32, num_classes=1,no_of_channels_=5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv3d(1, base_filters*(factor**0), kernel_size=(3,3,3)) #### 1,32
        self.conv2 = nn.Conv3d(base_filters*(factor**0), base_filters*(factor**2), kernel_size=(3,3,3)) #### 32,64
        self.conv3 = nn.Conv3d(base_filters*(factor**2), base_filters*(factor**3), kernel_size=(3,3,3)) #### 64,128
        self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(256 * no_of_channels_ * 14 * 14, 512) 
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x.view(x.shape[0],x.shape[1],x.shape[2],x.shape[-2],x.shape[-1]))))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 256 * x.shape[2] * 14 * 14)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define dataset class
class CustomDataset(Dataset):
    def __init__(self, data_dir, csv_file, target_size=(128, 128), seed_=False, testing=False, vis=False):
        self.data_dir           = data_dir
        self.file_list          = os.listdir(data_dir)
        self.labels             = pd.read_csv(csv_file)
        self.target_size        = target_size
        self.all_img_shapes     =   []
        self.check_all_img_sizes()
        self.seed               = seed_
        self.testing            = testing
        self.vis                = vis
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        try: file_ids = int(filename.split('fm0')[-1].split('_3d_recon.nii')[0])
        except: 
            del self.file_list[idx]
        if filename.endswith('.nii'):
            age = self.labels['tag_ga'][np.where(self.labels['ids']==file_ids)[0][0]]
            image_path = os.path.join(self.data_dir, filename)
            image = self.read_nifti_file(image_path)
            image = self.resize_image(image, self.target_size)
            if image is None or age is None:
                print("Data corrupted")
            return image, age

    def check_all_img_sizes(self):
        
        for filepath in self.file_list:
            image = nib.load(os.path.join(self.data_dir,filepath)).get_fdata()
            self.all_img_shapes.append(image.shape[0])
        return self.all_img_shapes
    
    def read_nifti_file(self, filepath):
        image = nib.load(filepath).get_fdata()
        return image

    def resize_image(self, image, target_size):
        resize_transform = TF.resize
        image_resized = []
        all_image_sizes = self.check_all_img_sizes()
        
        rng                     =   default_rng()
        max_size                =   60 # np.max(self.all_img_shapes)

        no_of_non_zero_perc     =   [(np.sum(image[index,:,:]>0)/(image.shape[-1]*image.shape[-2])) for index in range(image.shape[0])]
        no_of_non_zero_perc     =   np.array(no_of_non_zero_perc)
        index_roi               =   np.argsort(no_of_non_zero_perc)  # (np.where(no_of_non_zero_perc>(np.max(no_of_non_zero_perc)*0.2)))[0] # np.argsort(no_of_non_zero_perc)  # (np.where(no_of_non_zero_perc>(np.max(no_of_non_zero_perc)*0.3)))[0]
        """ #if self.seed:
            #np.random.seed(1000)
            #step  = image.shape[0]/max_size
            #numbers_chanel = [round(channel) for channel in np.linspace(0,image.shape[0]-1,max_size)] #
            #assert len(np.unique([round(channel) for channel in np.linspace(0,image.shape[0]-1,max_size)]))== max_size
            #numbers_chanel =    np.random.randint(low=0,high=image.shape[0],size=max_size)
            #rng.choice(image.shape[0], size=np.min(self.all_img_shapes) , replace=False)  #np.arange(60) #rng.choice(image.shape[0], size=np.min(self.all_img_shapes) , replace=False) ## image.shape[0]
            # if image.shape[0]<max_size and self.seed:
            #     no_of_additionl_planes  =   int(max_size - image.shape[0])
            #     append_numbers_channel  =   np.arange(no_of_additionl_planes)
            #     numbers_channel_total   =   np.concatenate((numbers_chanel,append_numbers_channel)) """
        
        if len(index_roi)>=max_size:
            
            numbers_chanel          =   index_roi[0:60] # np.random.choice(index_roi, max_size,replace=False)
            numbers_channel_total   =   numbers_chanel
        elif len(index_roi)<max_size:
            
            numbers_chanel          =   index_roi[0:len(index_roi)] #np.random.choice(index_roi, len(index_roi),replace=False)
            no_of_additionl_planes  =   int(max_size - len(index_roi))
            if not self.seed or no_of_additionl_planes>len(index_roi):
                append_numbers_channel  =   np.random.choice(index_roi, no_of_additionl_planes,replace=False)
            else:
                append_numbers_channel  =   index_roi[-no_of_additionl_planes:]
            numbers_channel_total       =   np.concatenate((numbers_chanel,append_numbers_channel))
        
        if self.vis:
            ncols = 10
            nrows = math.ceil(image.shape[0]/ncols)
            fig, axes = plt.subplots(nrows, ncols)
            row = 0 
            col = 0
            for data_indx in range(image.shape[0]):
                
                axes[row,col].imshow(image[data_indx, :, :])
                col+=1
                if col==10: 
                    col = 0
                    row+=1
        """ else:
            numbers_chanel          =   np.random.randint(low=0, high=image.shape[0],size=max_size)      #rng.choice(image.shape[0], size=np.min(self.all_img_shapes) , replace=False) ## image.shape[0]

        if len(numbers_chanel)<max_size: # and not self.seed:
            no_of_additionl_planes  =   int(max_size - image.shape[0])
            np.random.seed(2000)
            append_numbers_channel  =   np.random.randint(low=0,high=image.shape[0],size=no_of_additionl_planes) #rng.choice(image.shape[0], size=no_of_additionl_planes, replace=True)
            numbers_channel_total   =   np.concatenate((numbers_chanel,append_numbers_channel))

        else:
            numbers_channel_total   =   numbers_chanel """
            

        #numbers_channel_total       =   numbers_chanel #np.concatenate((numbers_chanel,append_numbers_channel))

        for slice_idx in numbers_channel_total[0:60]: #range(np.min(self.all_img_shapes)): #image.shape[0]
            add_augmentaion = np.random.choice([0,1],1)[0]
            slice_image = image[slice_idx, :, :]
            slice_image_resized = resize_transform(TF.to_pil_image(slice_image), target_size)
            if add_augmentaion==0 or self.testing: image_resized.append(TF.to_tensor(slice_image_resized))
            else: image_resized.append(aug_data(TF.to_tensor(slice_image_resized)))

        assert len(image_resized)==max_size
        return torch.stack(image_resized)

# Visualize dataset
def visualize_dataset(dataset):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(dataset)):
        try:        
            image, age = dataset[i]
            ax.scatter(age, image.shape[1], image.shape[2], marker='o', color='b')
        except: 
            pass #del dataset[i]
    ax.set_xlabel('Age')
    ax.set_ylabel('Height')
    ax.set_zlabel('Width')
    plt.title('Dataset Visualization')
    plt.show()
