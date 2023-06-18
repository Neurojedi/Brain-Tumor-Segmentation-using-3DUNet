# This code is based on the assignments from the Deep Learning for Medicine Course on Coursera. I have made modifications to the code to suit my specific requirements and added additional functions.
import nilearn as nl
import nibabel as nib
import numpy as np
import h5py
import os
import json
from sklearn.model_selection import train_test_split
from tensorflow import keras
from util import load_case
def get_sub_volume(image, label, 
                   orig_x = 240, orig_y = 240, orig_z = 155, 
                   output_x = 160, output_y = 160, output_z = 16,
                   num_classes = 4, max_tries = 1500, 
                   background_threshold=0.95):

    # Initialize features and labels with `None`
    X = None
    y = None
    tries = 0
    
    while tries < max_tries:
        # randomly sample sub-volume by sampling the corner voxel
        start_x = np.random.randint(0 , orig_x - output_x + 1)
        start_y = np.random.randint(0 , orig_y - output_y + 1)
        start_z = np.random.randint(0 , orig_z - output_z + 1)

        # extract relevant area of label
        y = label[start_x: start_x + output_x,
                  start_y: start_y + output_y,
                  start_z: start_z + output_z]
        
        # One-hot encode the categories by adding a 4th dimension, 'num_classes': (output_x, output_y, output_z, num_classes)
        y = keras.utils.to_categorical(y, num_classes = num_classes)

        
        bgrd_ratio = np.sum(y[:,:,:,0]) / (output_x * output_y * output_z) # compute the background ratio
        tries += 1 # increment tries counter

        # if background ratio is below the desired threshold, use that sub-volume.
        # otherwise continue the loop and try another random sub-volume
        if bgrd_ratio < background_threshold:
            X = np.copy(image[start_x: start_x + output_x,
                              start_y: start_y + output_y,
                              start_z: start_z + output_z, :])
            
            X = np.moveaxis(X,3,0)
            y = np.moveaxis(y,3,0)
            y = y[1:, :, :, :]  
            return X, y
    # if we've tried max_tries number of samples, break to avoid looping forever.
    #print(f"Tried {tries} times to find a sub-volume. Giving up...")

def standardize(image):
    
    # initialize to array of zeros, with same shape as the image
    standardized_image = np.zeros(image.shape)

    # iterate over channels
    for c in range(image.shape[0]):
        # iterate over the `z` dimension
        for z in range(image.shape[3]):
            image_slice = image[c,:,:,z]

            centered = image_slice - np.mean(image_slice)
            centered_scaled = centered / np.std(centered)

            # update  the slice of standardized image
            standardized_image[c, :, :, z] = centered_scaled

    return standardized_image
    
from tqdm import tqdm
def CreateSubVolumes(jsonfile_path, outputpath):
  isExist = os.path.exists(outputpath)
  if not isExist:
    os.makedirs(outputpath)
  xtrain_imageslist=[]
  ytrain_imageslist=[]
  
  with open(jsonfile_path) as json_file:
    dset = json.load(json_file)
  
  print("Creating the training list...")
  for i in tqdm(range(len(dset["training"])),colour="blue"):
    xtrain_imageslist.append(dset["training"][i]["image"])
    ytrain_imageslist.append(dset["training"][i]["label"])
  print("\n")
  print("Creating the sub-volumes... ")
  nosubvolumes=[]
  for i in tqdm(range(len(dset["training"])),colour="red"):
     image, label = load_case("/content/Task01_BrainTumour/"+ xtrain_imageslist[i], "/content/Task01_BrainTumour/"+ ytrain_imageslist[i])
     try:
        X, y = get_sub_volume(image, label)
     except TypeError:
       nosubvolumes.append(xtrain_imageslist[i])
       continue;
     X_norm = standardize(X)
     y_arranged = np.moveaxis(y, [0,1,2,3], [3,0,1,2])
     hf = h5py.File(outputpath+xtrain_imageslist[i].split("/")[2]+".h5", 'w')
     hf.create_dataset('x', data=X_norm)
     hf.create_dataset('y', data=y_arranged)
     hf.close()
  print("\n")
  jsonFile = open("nosubvolumeslist.json", "w")
  nsub=json.dumps(nosubvolumes)
  jsonFile.write(nsub)
  jsonFile.close()
  print(f"For {len(nosubvolumes)} files, could not find a sub-volume after trying 1000 times.")


def get_config(h5_path, test_size):
  h5_list=[]
  for path in os.listdir(h5_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(h5_path, path)):
        h5_list.append(path)
  train,test=train_test_split(h5_list,test_size=test_size,random_state=1777) # Gauss :)
  g = {
  "train": train,
  "valid": test
  }
  y = json.dumps(g)
  jsonFile = open("config.json", "w")
  jsonFile.write(y)
  jsonFile.close()
