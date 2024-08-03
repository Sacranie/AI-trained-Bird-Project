import os 
import cv2
import shutil
from sklearn.model_selection import train_test_split
from bing_image_downloader import downloader
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np



#Step 1 is to upload images of each bird from bing onto seperate directories. In order to do this we need to ensure that the directories
#actually exist. We also need a list of the bird and use the download function on python to direct this into the right output directory

birds = [
    "Sparrow", "Pigeon", "Crow", "Robin", "Cassowary", "Emu"
]
 # Use a list instead of a set to maintain order

# Base directory to store images
base_dir = r"C:\Husayn Program\python\Bird_Project\non_processed"

#This for loop essentially iterates through each of the type of birds, makes a seperate direcctory for them in the non processed directory
#and downloads 10 images of each
for bird in birds:
    bird_dir = os.path.join(base_dir, bird) #This bird directory will be ...\non_processed\Sparrow for example.

    downloader.download(bird, limit=100, output_dir=base_dir) #This downloads the images and puts them into the folders

#Step 2 is to create a fnction which processes the images and puts them into a seperate directory known as processed
def preprocess_images(input_dir, output_dir, image_size=(128, 128)):
    """
    The Purpose of this is to process the images so it is easier for the model to read it 
    The greyscale conversion simplifies the data. This reduces the complexity of the data 
    and colour is not required for distinguishing between these classes.

    Parameters:
    - input_dir (str): Path to the directory containing images.
    - output_dir (str): Path to the directory where the image after being resized and greyscaled will be saved.
    - image_size: The size of the output image required to be 
    Returns:
    - None: Images are saved in the specified output directory structure.
    """

    # Iterate over each label (subdirectory) in the input directory
    #The input directory will be the non processed images]
    for label in os.listdir(input_dir):
        label_dir = os.path.join(input_dir, label)  # Full path to the current label directory. This will be...\non_procssed\Sparrow
        output_label_dir = os.path.join(output_dir, label)  # Full path to the corresponding output label directory. This will be ...\processed\Sparrow
        
        # Create the corresponding output subdirectory if it doesn't exist. So important to make sure the directory actually exists.
        if not os.path.exists(output_label_dir):
            os.makedirs(output_label_dir)
        
        # Iterate over each image file in the current label directory
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)  # Full path to the current image file
            
            # Read the image NOT in grayscale mode. Keeping it in colour so we can distinguish between the different birds easily.
            img = cv2.imread(img_path)
                
            # If the image is successfully read, process it
            if img is not None:
                # Resize the image to the specified size
                img_resized = cv2.resize(img, image_size)
                    
                # Save the processed image to the corresponding output directory. The final output directory will be  ...\processed\Sparrow\Image
                cv2.imwrite(os.path.join(output_label_dir, img_name), img_resized)

#Step 3 is to split our images into train validation and test data set for our AI model to learn from in Step 4.
def split_dataset(input_dir, output_dir, test_size=0.3, val_size=0.3):
    """
    Splits images in input_dir into train, validation, and test sets, 
    and saves them in the specified output_dir.

    Parameters:
    - input_dir (str): Path to the directory containing subdirectories of images.
    - output_dir (str): Path to the directory where split datasets will be saved.
    - test_size (float or int, optional): Proportion of the dataset to include in the test split.
      Default is 0.2 (20%).
    - val_size (float or int, optional): Proportion of the training set to include in the validation split.
      Default is 0.2 (20% of the training set).

    Returns:
    - None: Datasets are saved in the specified output directory structure.
    """
    # Create output directory if it doesn't exist. So this is a serperate dataset directory. This is still important as this is
    # ...\dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Iterate over each label (subdirectory) in the input directory. This goes over each bird file
    for label in os.listdir(input_dir):
        label_dir = os.path.join(input_dir, label)
        
        # List all images in the current label directory. This means we label all the image for each bird e.g all the images for Sparrow
        images = os.listdir(label_dir)
        
        # Split images into train and test sets
        # This means 80 percent of the images are allocates to training the model
        train_images, test_images = train_test_split(images, test_size = test_size)
        
        # Further split train set into train and validation sets
        train_images, val_images = train_test_split(train_images, test_size = val_size)
        
        # Iterate over categories (train, val, test)
        for category, image_list in zip(['train', 'val', 'test'], [train_images, val_images, test_images]):
            category_dir = os.path.join(output_dir, category, label) #So this is...\dataset\train\Sparrow for example
            
            # Create category directory if it doesn't exist
            if not os.path.exists(category_dir):
                os.makedirs(category_dir)
            
            # Copy each image to its corresponding category directory
            for image in image_list:
                shutil.copy(os.path.join(label_dir, image), os.path.join(category_dir, image))

# Directories
base_dir = r"C:\Husayn Program\python\Bird_Project"
processed_dir = os.path.join(base_dir, "Processed")
dataset_dir = os.path.join(base_dir, "Dataset")

# Preprocess images
preprocess_images(os.path.join(base_dir, "non_processed"), processed_dir)

# Split dataset
split_dataset(processed_dir, dataset_dir)

# Image Data Generators
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'val')
test_dir = os.path.join(dataset_dir, 'test')

# The purpose of the next line of codes is to resize the pixels so they are tbeween 0-1.
# Normalising it makes it easier for the model to read and process it.
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# The train generator has 4 inputs. The batch size refers to the number of samples that will be propogated trhough the network at once 
# This is more efficient and leds to faster training.
# The class mode refers to the type of label arrays that are returned
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(birds), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Save the model
model.save(os.path.join(base_dir, 'bird_classifier_model.keras'))


# Load the trained model
model = load_model(os.path.join(base_dir, 'bird_classifier_model.keras'))

def predict_bird(image_path):
    img = cv2.imread(image_path)  # Read image without specifying grayscale mode
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    img = cv2.resize(img, (128, 128))  # Resize to 128x128
    img = img / 255.0  # Normalize pixel values
    
    # Ensure the image has 3 channels if your model expects RGB input
    if img.shape[-1] != 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB if necessary
    
    img = img.reshape(1, 128, 128, 3)  # Reshape for model input (batch_size, height, width, channels)
    
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)
    
    return birds[predicted_class[0]]

# Example usage
image_path = r"C:\Husayn Program\python\Bird_Project\emu2.jpg"
predicted_bird = predict_bird(image_path)
print(f"The predicted bird is: {predicted_bird}")