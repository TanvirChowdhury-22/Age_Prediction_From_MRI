import torch
from torchvision.models import resnet18
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torch import nn 
import numpy as np 
import cv2
import torch.nn.functional as f
from torch.utils.data import Dataset
import nibabel as nib
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from numpy.random import default_rng
import math
# Define constants
from torchvision.models import ResNet18_Weights
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

from torchvision import transforms


# Define your transformations for data augmentation
def aug_data(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy array to PIL Image  
        transforms.RandomResizedCrop(size=(128, 128),scale=(0.975,0.975)), # randomly resize      
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomVerticalFlip(),    # Randomly flip the image vertically
        transforms.RandomRotation(90),   # Randomly rotate the image by up to 90 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Randomly jitter brightness and contrast
        transforms.ToTensor()           # Convert PIL Image back to tensor
    ]) # transforms.RandomResizedCrop(size=(128, 128),scale=(0.975,0.975))

    return transform(image)


class Resnet50WithFPN(torch.nn.Module):
    def __init__(self):
        super(Resnet50WithFPN, self).__init__()
        # Get a resnet50 backbone
        model           = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # check the available nodes
        """ train_nodes, eval_nodes = get_graph_node_names(model()) 
        return_nodes = {
            'layer1': 'layer1',
            'layer2': 'layer2',
            'layer3': 'layer3',
            'layer4': 'layer4',
        } """ ### node name: user-specified key for the output dict
        
        self.body           = create_feature_extractor(model, return_nodes={f'layer{k}': str(v)
                             for v, k in enumerate([1, 2, 3, 4])})
        
        self.fully_connected_size   = 122880
        self.fully_connected        = nn.Linear(self.fully_connected_size, 1)
        # Dry run to get number of channels for FPN
        """ inp = torch.randn(120, 3, 224, 224)
        with torch.no_grad():
            out = self.body(inp)
        in_channels_list = [o.shape[1] for o in out.values()] """
        # Build FPN
        #self.out_channels = 256
        """ self.fpn = FeaturePyramidNetwork(
            in_channels_list, out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool()) """
    
    
    def forward(self, x):
        with torch.no_grad():
            #x       = self.pre_processing(x)
            x        = self.body(x) ### return features extraction 
            x        = torch.concatenate((torch.mean(x['0'],axis=0).flatten(), torch.mean(x['1'],axis=0).flatten(), torch.mean(x['2'],axis=0).flatten(), torch.mean(x['3'],axis=0).flatten()))
            #self.fully_connected_size   =   x.shape[0]
        
        #x           = f.leaky_relu(self.fully_connected)
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
        #self.feature_ex         = model_fx
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

    """ def pre_processing(self,inp):
        #inp           = np.random.randint(low=0, high=255, size = (120,128, 128,1)).astype(np.uint8)
        inp_tensor    = torch.zeros((inp.shape[0],3, inp.shape[1], inp.shape[2]))
        for indx in range(inp.shape[0]):
            rgb_img          = (cv2.cvtColor(inp[indx][:,:,0],cv2.COLOR_GRAY2RGB))
            inp_tensor[indx] = torch.tensor(rgb_img).permute(2,0,1)/torch.max(inp_tensor)  
        return inp_tensor """
    
    def resize_image(self, image, target_size):
        resize_transform = TF.resize
        image_resized = []
        no_of_non_zero_perc     =   [(np.sum(image[index,:,:]>0)/(image.shape[-1]*image.shape[-2])) for index in range(image.shape[0])]
        no_of_non_zero_perc     =   np.array(no_of_non_zero_perc)
        index_roi               =   (np.where(no_of_non_zero_perc>(np.max(no_of_non_zero_perc)*0.8)))[0]

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


        #numbers_channel_total       =   numbers_chanel #np.concatenate((numbers_chanel,append_numbers_channel))
        feature_ex              = Resnet50WithFPN()
        for slice_idx in index_roi: #range(image.shape[0]): #range(np.min(self.all_img_shapes)): #image.shape[0]
            add_augmentaion     = np.random.choice([0,1],1)[0]
            if np.max(image[slice_idx, :, :])>255: image_ = ((image[slice_idx, :, :]/np.max(image[slice_idx, :, :]))*255).astype(np.uint8)
            else: image_        = image[slice_idx, :, :].astype(np.uint8)
            slice_image         = cv2.cvtColor(image_ ,cv2.COLOR_GRAY2RGB)
            #slice_image         = torch.tensor(slice_image).permute(2,0,1).numpy() # 
            slice_image_resized = resize_transform(TF.to_pil_image(slice_image), target_size)
            if add_augmentaion==0 or self.testing: image_resized.append(TF.to_tensor(slice_image_resized))
            else: image_resized.append(aug_data(TF.to_tensor(slice_image_resized)))

        #assert len(image_resized)==max_size
        return feature_ex(torch.stack(image_resized))

class FNN(nn.Module):
    def __init__(self, factor=2, base_filters=32, num_classes=1,no_of_channels_=5):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(122880, 512) 
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
""" model           = Resnet50WithFPN()
input           = np.random.randint(low=0, high=255, size = (120,128, 128,1)).astype(np.uint8)
output_features = model(input) """