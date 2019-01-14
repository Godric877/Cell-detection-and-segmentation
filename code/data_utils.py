from __future__ import print_function, division
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageDraw, ImageColor

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import pprint
import string
import random

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# interactive mode
plt.ion() 

categories = set()
categories = {'gametocyte', 'schizont', 'trophozoite', 'difficult', 'leukocyte', 'ring', 'red blood cell'}
colors = {
# https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
  'red blood cell': (230, 25, 75),   # Red
  'schizont': (60, 180, 75),         # Green
  'trophozoite': (255, 225, 25),     # Yellow
  'difficult': (0, 0, 0),            # Black
  'leukocyte': (70, 240, 240),       # Cyan
  'ring': (240, 50, 230),            # Orange
  'gametocyte': (240, 50, 230)       # Magenta
}

def draw_rect(image, start, end, type='red blood cell'):
#   Draws rectangle specified by corners
#    
# Params:
#       image (PIL Image)
#       start (int, int): top left corner
#       end (int, int): bottom right corner
  draw = ImageDraw.Draw(image)
  color = colors[type]
  draw.line([start[0], start[1], start[0], end[1]], fill=color, width=5)
  draw.line([start[0], end[1], end[0], end[1]], fill=color, width=5)
  draw.line([end[0], end[1], end[0], start[1]], fill=color, width=5)
  draw.line([end[0], start[1], start[0], start[1]], fill=color, width=5)
  del draw

def get_coordinates(box):
# Get coordinates of corners of bounding box
  return (box['bounding_box']['minimum']['c'],\
    box['bounding_box']['minimum']['r']),\
    (box['bounding_box']['maximum']['c'],\
    box['bounding_box']['maximum']['r'])

def show_bounding_boxes(image,bounding_boxes):
#   Displays image with given bounding boxes
#    
#   Params:
#     Numpy array of image, bounding_boxes object
#   Returns:
#     PIL Image
  img = Image.fromarray(image.astype('uint8'), 'RGB')

  for obj in bounding_boxes:
    start, end = get_coordinates(obj)
    draw_rect(img, start, end, type=obj['category'])
  
  img.show()

def show_patches_batch(batch_sample,num_patches=200,save_image=False):
#   Show image for a batch of samples.
    patches_batch = batch_sample['patches']
    grid = utils.make_grid(patches_batch[0:num_patches,:,:,:],nrow=20)
    img_array = grid.numpy().transpose((1, 2, 0))
    plt.figure()
    plt.imshow(img_array)
    plt.title('Batch from dataloader')
    if(save_image):
      filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)) + ".png"
      utils.save_image(grid,filename=filename)
    plt.show()

class malaria_dataset(Dataset):
#   Malaria Cells dataset with bounding boxes

    def __init__(self, json_file, root_dir, transform=None):   
    # Params:
    #     json_file (string): Path to the json file with annotations.
    #     root_dir (string): Topmost directory with images and json files.
    #     transform (callable, optional): Optional transform to be applied
    #         on a sample.
        with open(json_file) as F:
          self.img_json = json.load(F)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_json)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir,
                                self.img_json[index]['image']['pathname'][1:])
        image = io.imread(img_name)
        bounding_boxes = self.img_json[index]['objects']
        sample = {'image': image/255., 'bounding_boxes': bounding_boxes}

        if self.transform:
            sample = self.transform(sample)
         
        return sample

class crop_bounding_box(object):
#   Transformation class to crop each image at the bounding boxes and resize to output size  
#   Params:
#     output_size : tuple specifying size of each patch
    def __init__(self,output_size):
      self.output_size = output_size

    def __call__(self,sample):
      image, bounding_boxes = sample['image'],sample['bounding_boxes']
      img = Image.fromarray(image.astype('uint8'), 'RGB')
      patches = []
      for obj in bounding_boxes:
        start, end = get_coordinates(obj)
        patch = image[start[1]:end[1],start[0]:end[0]]
        patch = transform.resize(patch, self.output_size)
        patch = patch.transpose((2, 0, 1))
        patch = torch.from_numpy(patch)
        patches.append(patch)
      image = image.transpose((2, 0, 1))
      image = torch.from_numpy(image)
      return {'image' : image,'bounding_boxes' : bounding_boxes,'patches' : patches}

def collate_fn(data):
#   Collate mini-batches for data loader
#   Params:
#     list of data with size = batch_size
#   Returns:
#     dict with collated data
    images_batch = []
    patches_batch = []
    bounding_boxes_batch = []

    for i_batch,batch_sample in enumerate(data):
      images_batch.append(data[i_batch]['image'])
      patches_batch.extend(data[i_batch]['patches'])
      bounding_boxes_batch.append(data[i_batch]['bounding_boxes'])

    patches_batch = torch.stack(patches_batch,0)
        
    return {'image' : images_batch, 'patches' : patches_batch, 'bounding_boxes' : bounding_boxes_batch}