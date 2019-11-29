# Earth_Images_Dataset
This repository contains a dataset of Earth Images generated using Systems Tool Kit (STK). 
The images are generated for a 600 km altitude, Equatorial Low Earth Orbit. 
They are divided into 10 classes depending on the longitude from where they have been taken. Thus, one image at a step of 36 degrees. 
The dataset contains 100 training images and 40 test images. 

The dataset can be accessed in google colab using - 
! git clone https://github.com/anant-infinity/Earth_Images_Dataset.git
! ls

1. main_file.py is the code to run. This loads the data set and also calls the run network function to run the nueral network
2. network_class.py contains the class defination of the nueral network which also implements different optimizers 
3. data_augmentation can be used to augment data by rotation , skewing , shifting and changing brightness 


