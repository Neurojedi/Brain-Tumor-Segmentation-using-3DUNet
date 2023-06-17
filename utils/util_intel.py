# Adapted from https://github.com/IntelAI/unet
import json
import shutil
import tqdm
import tensorflow as tf
from tensorflow import keras as K

import nibabel as nib
import numpy as np
import os
import datetime

crop_dim = (128,128,128,1)
number_output_classes = 3

filters=8
saved_model_name = "3d_unet_decathlon"

def seperate_subvolumes(jsonfile_path,output_path):
  with open(jsonfile_path) as json_file:
    dset = json.load(json_file)
  isExist = os.path.exists(output_path)
  if not isExist:
    os.makedirs(output_path)
  for i in tqdm.tqdm(dset,colour="green"):
    shutil.move('/content/Task01_BrainTumour/'+str(i), output_path)

class DatasetGenerator:
        
    def __init__(self, data_path):
        
        self.data_path = data_path
        self.create_file_list()

    def create_file_list(self):
        """
        Get list of the files from the BraTS raw data
        Split into training and testing sets.
        """
        import os
        import json
        
        json_filename = os.path.join(self.data_path, "dataset.json")

        try:
            with open(json_filename, "r") as fp:
                experiment_data = json.load(fp)
        except IOError as e:
            print("File {} doesn't exist. It should be part of the "
                  "Decathlon directory".format(json_filename))

        self.output_channels = experiment_data["labels"]
        self.input_channels = experiment_data["modality"]
        self.description = experiment_data["description"]
        self.name = experiment_data["name"]
        self.release = experiment_data["release"]
        self.license = experiment_data["licence"]
        self.reference = experiment_data["reference"]
        self.tensorImageSize = experiment_data["tensorImageSize"]
        self.numFiles = experiment_data["numTraining"]
        
        """
        Create a dictionary of tuples with image filename and label filename
        """
        self.filenames = {}
        for idx in range(self.numFiles):
            self.filenames[idx] = [os.path.join(self.data_path,
                                              experiment_data["training"][idx]["image"]),
                                    os.path.join(self.data_path,
                                              experiment_data["training"][idx]["label"])]
            
        
    def print_info(self):
        """
        Print the dataset information
        """

        print("="*30)
        print("Dataset name:        ", self.name)
        print("Dataset description: ", self.description)
        print("Tensor image size:   ", self.tensorImageSize)
        print("Dataset release:     ", self.release)
        print("Dataset reference:   ", self.reference)
        print("Input channels:      ", self.input_channels)
        print("Output labels:       ", self.output_channels)
        print("Dataset license:     ", self.license)
        print("="*30)

data_path="/content/Task01_BrainTumour/"
brats_datafiles = DatasetGenerator(data_path)

def z_normalize_img(img):
    """
    Normalize the image so that the mean value for each image
    is 0 and the standard deviation is 1.
    """
    for channel in range(img.shape[-1]):

        img_temp = img[..., channel]
        img_temp = (img_temp - np.mean(img_temp)) / np.std(img_temp)

        img[..., channel] = img_temp

    return img
    
def crop(img, msk, randomize):
        """
        Randomly crop the image and mask
        """

        slices = []
        
        # Do we randomize?
        is_random = randomize and np.random.rand() > 0.5

        for idx in range(len(img.shape)-1):  # Go through each dimension

            cropLen = crop_dim[idx]
            imgLen = img.shape[idx]

            start = (imgLen-cropLen)//2

            ratio_crop = 0.20  # Crop up this this % of pixels for offset
            # Number of pixels to offset crop in this dimension
            offset = int(np.floor(start*ratio_crop))

            if offset > 0:
                if is_random:
                    start += np.random.choice(range(-offset, offset))
                    if ((start + cropLen) > imgLen):  # Don't fall off the image
                        start = (imgLen-cropLen)//2
            else:
                start = 0

            slices.append(slice(start, start+cropLen))

        return img[tuple(slices)], msk[tuple(slices)]
    
def augment_data(img, msk, crop_dim):
    """
    Data augmentation
    Flip image and mask. Rotate image and mask.
    """
    
    # Determine if axes are equal and can be rotated
    # If the axes aren't equal then we can't rotate them.
    equal_dim_axis = []
    for idx in range(0, len(crop_dim)):
        for jdx in range(idx+1, len(crop_dim)):
            if crop_dim[idx] == crop_dim[jdx]:
                equal_dim_axis.append([idx, jdx])  # Valid rotation axes
    dim_to_rotate = equal_dim_axis

    if np.random.rand() > 0.5:
        # Random 0,1 (axes to flip)
        ax = np.random.choice(np.arange(len(crop_dim)-1))
        img = np.flip(img, ax)
        msk = np.flip(msk, ax)

    elif (len(dim_to_rotate) > 0) and (np.random.rand() > 0.5):
        rot = np.random.choice([1, 2, 3])  # 90, 180, or 270 degrees

        # This will choose the axes to rotate
        # Axes must be equal in size
        random_axis = dim_to_rotate[np.random.choice(len(dim_to_rotate))]
        
        img = np.rot90(img, rot, axes=random_axis)  # Rotate axes 0 and 1
        msk = np.rot90(msk, rot, axes=random_axis)  # Rotate axes 0 and 1

    return img, msk
    
def read_nifti_file(idx, crop_dim, randomize=False):
    """
    Read Nifti file
    """
    
    idx = idx.numpy()
    imgFile = brats_datafiles.filenames[idx][0]
    mskFile = brats_datafiles.filenames[idx][1]
    
    img = np.array(nib.load(imgFile).dataobj)
    
    img = np.rot90(img[...,[0]]) # Just take the FLAIR channel (0)
    
    msk = np.rot90(np.array(nib.load(mskFile).dataobj))

    """
    "labels": {
         "0": "background",
         "1": "edema",
         "2": "non-enhancing tumor",
         "3": "enhancing tumour"}
     """
    # Combine all masks but background
    if number_output_classes == 1:
        msk[msk > 0] = 1.0
        msk = np.expand_dims(msk, -1)
    else:
        msk_temp = np.zeros(list(msk.shape) + [number_output_classes])
        for channel in range(number_output_classes):
            msk_temp[msk==channel,channel] = 1.0
        msk = msk_temp
    
    imgFilename = (os.path.basename(brats_datafiles.filenames[idx][0])).split(".nii.gz")[0]
    
    # Crop
    img, msk = crop(img, msk, randomize)
    
    # Normalize
    img = z_normalize_img(img)
    
    # Randomly rotate
    if randomize:
        img, msk = augment_data(img, msk, crop_dim)
    
    return img, msk


def unet_3d(fms=32, input_dim=crop_dim, use_upsampling=False, concat_axis=-1):
    """
    3D U-Net
    """
    
    def ConvolutionBlock(x, name, fms, params):
        """
        Convolutional block of layers
        Per the original paper this is back to back 3D convs
        with batch norm and then ReLU.
        """

        x = K.layers.Conv3D(filters=fms, **params, name=name+"_conv0")(x)
        x = K.layers.BatchNormalization(name=name+"_bn0")(x)
        x = K.layers.Activation("relu", name=name+"_relu0")(x)

        x = K.layers.Conv3D(filters=fms, **params, name=name+"_conv1")(x)
        x = K.layers.BatchNormalization(name=name+"_bn1")(x)
        x = K.layers.Activation("relu", name=name)(x)

        return x

    inputs = K.layers.Input(shape=input_dim, name="MRImages")

    params = dict(kernel_size=(3, 3, 3), activation=None,
                  padding="same", 
                  kernel_initializer="he_uniform")

    # Transposed convolution parameters
    params_trans = dict(kernel_size=(2, 2, 2), strides=(2, 2, 2),
                        padding="same")


    # BEGIN - Encoding path
    encodeA = ConvolutionBlock(inputs, "encodeA", fms, params)
    poolA = K.layers.MaxPooling3D(name="poolA", pool_size=(2, 2, 2))(encodeA)

    encodeB = ConvolutionBlock(poolA, "encodeB", fms*2, params)
    poolB = K.layers.MaxPooling3D(name="poolB", pool_size=(2, 2, 2))(encodeB)

    encodeC = ConvolutionBlock(poolB, "encodeC", fms*4, params)
    poolC = K.layers.MaxPooling3D(name="poolC", pool_size=(2, 2, 2))(encodeC)

    encodeD = ConvolutionBlock(poolC, "encodeD", fms*8, params)
    poolD = K.layers.MaxPooling3D(name="poolD", pool_size=(2, 2, 2))(encodeD)

    encodeE = ConvolutionBlock(poolD, "encodeE", fms*16, params)
    # END - Encoding path

    # BEGIN - Decoding path
    if use_upsampling:
        up = K.layers.UpSampling3D(name="upE", size=(2, 2, 2),
                                   interpolation="bilinear")(encodeE)
    else:
        up = K.layers.Conv3DTranspose(name="transconvE", filters=fms*8,
                                      **params_trans)(encodeE)
    concatD = K.layers.concatenate(
        [up, encodeD], axis=concat_axis, name="concatD")

    decodeC = ConvolutionBlock(concatD, "decodeC", fms*8, params)

    if use_upsampling:
        up = K.layers.UpSampling3D(name="upC", size=(2, 2, 2),
                                   interpolation="bilinear")(decodeC)
    else:
        up = K.layers.Conv3DTranspose(name="transconvC", filters=fms*4,
                                      **params_trans)(decodeC)
    concatC = K.layers.concatenate(
        [up, encodeC], axis=concat_axis, name="concatC")

    decodeB = ConvolutionBlock(concatC, "decodeB", fms*4, params)

    if use_upsampling:
        up = K.layers.UpSampling3D(name="upB", size=(2, 2, 2),
                                   interpolation="bilinear")(decodeB)
    else:
        up = K.layers.Conv3DTranspose(name="transconvB", filters=fms*2,
                                      **params_trans)(decodeB)
    concatB = K.layers.concatenate(
        [up, encodeB], axis=concat_axis, name="concatB")

    decodeA = ConvolutionBlock(concatB, "decodeA", fms*2, params)

    if use_upsampling:
        up = K.layers.UpSampling3D(name="upA", size=(2, 2, 2),
                                   interpolation="bilinear")(decodeA)
    else:
        up = K.layers.Conv3DTranspose(name="transconvA", filters=fms,
                                      **params_trans)(decodeA)
    concatA = K.layers.concatenate(
        [up, encodeA], axis=concat_axis, name="concatA")

    # END - Decoding path

    convOut = ConvolutionBlock(concatA, "convOut", fms, params)

    prediction = K.layers.Conv3D(name="PredictionMask",
                                 filters=number_output_classes, kernel_size=(1, 1, 1),
                                 activation="sigmoid")(convOut)

    model = K.models.Model(inputs=[inputs], outputs=[prediction], name="3d_unet_decathlon")

    return model
