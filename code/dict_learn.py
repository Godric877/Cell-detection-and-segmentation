import data_utils
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import time

#####################################################
# Parameters

learning_rate_init = 3.7*1e-1 	# Learning rate for finding minimum
reg_init = 0 					          # Regularisation strength
num_iter_init = 1500 			      # Number of iterations
K = 15 							            # Value of K (Size of basis)
patch_size = 45 				        # Size of patches with centered cells
batch_size = 2  				        # Size of data being processed each iteration
D_in = patch_size*patch_size*3 	# Dimension of each patch = number of pixels in each patch
num_workers = 4					        # Number of workers

#####################################################
# Load Data

dtype = torch.double
device = torch.device("cpu")

train_data = data_utils.malaria_dataset(json_file='../datasets/malaria/training.json',
									    root_dir='../datasets/malaria',
									    transform=data_utils.crop_bounding_box((patch_size,patch_size)))

training_data_loader = torch.utils.data.DataLoader(train_data,
												   batch_size=batch_size,
												   shuffle=True,
												   num_workers=num_workers,
                           collate_fn=data_utils.collate_fn)

#####################################################
# Basis Vector Initialisation 

for i_batch, batch_sample in enumerate(training_data_loader):

    patches_batch = batch_sample['patches']
    num_patches = len(patches_batch)

    # Reshape sampled batch
    x = patches_batch.view(num_patches,-1)

    # Initialise w
    w = torch.randn(num_patches, num_patches, device=device, dtype=dtype, requires_grad=True)
    print(x.shape, x.mean(), w.mean())


    for epoch in range(num_iter_init):
      # Evaluate Loss
      # w = torch.nn.functional.normalize(w, p=2, dim=1)
      temp1 = x - w.mm(x) + w.diag()[:,None].expand_as(x)*x
      #print((temp1**2).mean())
      temp2 = temp1.view(num_patches*D_in).pow(2).mean()
      x_norm = (x**2).sum(1).view(-1, 1)
      y_norm = x_norm.view(1, -1)
      dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(x, 0, 1))
      dist_norm = torch.nn.functional.normalize(dist, p=2, dim=1)
      dist_exp = torch.exp(dist_norm)

      loss = temp2 + reg_init*((dist_exp * w).view(num_patches*num_patches).pow(2).sum())

      # Print loss and iteration
      print(epoch, loss.item())

      # Calculate Gradient
      loss.backward()

      # Update weights
      with torch.no_grad():
        w -= learning_rate_init * w.grad
        w.grad.zero_()

    if(i_batch == 0):
      break

w = torch.nn.functional.normalize(w, p=2, dim=1)
basis_index = w.sum(0).sort()[1][K:] # The index of elements selected as the basis for the dictionary
basis = x[basis_index] # Basis Initialisation
coeff = (w.clone().t()[basis_index]).t() # The coefficients corresponding to the initialised basis

######################################################
