# Importing necessary functions 
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img 
   
# Initialising the ImageDataGenerator class. 
# We will pass in the augmentation parameters in the constructor. 
datagen = ImageDataGenerator( 
        rotation_range = 2, 
        shear_range = 0.2, 
        zoom_range = 0.2, 
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip = False, 
        brightness_range = (0.5, 1.5)) 
    
# Loading a sample image  
img = load_img('/content/Earth_Images_Dataset/train/long_108/1.jpg')  
# Converting the input sample image to an array 
x = img_to_array(img) 
# Reshaping the input image 
x = x.reshape((1, ) + x.shape)  
   
# Generating and saving 5 augmented samples  
# using the above defined parameters.  
i = 0
num_images_to_generate = 96
for batch in datagen.flow(x, batch_size = 1, 
                          save_to_dir ='/content/Earth_Images_Dataset/train/long_108',  
                          save_prefix ='image', save_format ='jpeg'): 
    i += 1
    if i > num_images_to_generate-1: 
        break
