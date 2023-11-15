# Brain Tumor Segmentation using 3D U-Net (Under Construction 🛠️)

<img src="tumorcombined.gif"></img>

This repository contains code for training two different U-Net based architectures using various loss functions and optimizers on the [Medical Segmentation Decathlon](http://medicaldecathlon.com/) dataset. The first model is an adaptation of an assignment from the [AI for Medicine Specialization on Coursera](https://www.coursera.org/specializations/ai-for-medicine), where we performed brain tumor segmentation using a small portion of the dataset. I have taken the pre-trained weights used in the assignment, expanded and improved the utility functions to enable training on the entire dataset. The model was trained either with pre-trained weights or from scratch, and during training, I employed both Adam and Ranger Optimizers. Additionally, I wanted to further evaluate loss functions, so I trained another 3D U-Net from the [Deep Learning Medical Decathlon Demos for Python](https://github.com/IntelAI/unet/tree/master) repository. This model was trained from scratch using a different train/validation split setting but with the same optimizer and loss function settings.


For more information about the dataset and data analysis, please refer to the Main.ipynb file. This notebook provides details about the architectures, training settings, and the results on the test set for comparison of choice in loss function and optimizer. You can also find a link to the Google Drive Folder, where all the weight files generated during training are stored.

Furthermore, you can explore the training settings I used in the notebooks organized under specific folders named after each respective loss function. Each notebook in these folders displays the training plot, and if you wish to retrain the network in a different setting, you can easily utilize the notebooks on Colab.










