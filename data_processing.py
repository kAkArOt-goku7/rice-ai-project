# data_processing.py
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def explore_dataset(data_path):
    """
    Explore the dataset - count images in each category
    """
    print("Exploring dataset structure...")
    
    classes = os.listdir(data_path)
    print(f"Found {len(classes)} classes: {classes}")
    
    for class_name in classes:
        class_path = os.path.join(data_path, class_name)
        if os.path.isdir(class_path):
            num_images = len(os.listdir(class_path))
            print(f"  {class_name}: {num_images} images")
    
    return classes

def load_and_preprocess_data(data_path, img_size=(224, 224)):
    """
    Load images and prepare them for training
    """
    print("Loading and preprocessing images...")
    
    images = []
    labels = []
    class_names = os.listdir(data_path)
    
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_path, class_name)
        if not os.path.isdir(class_path):
            continue
            
        print(f"Processing {class_name}...")
        
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Load image
                img_path = os.path.join(class_path, img_file)
                img = tf.keras.utils.load_img(img_path, target_size=img_size)
                img_array = tf.keras.utils.img_to_array(img)
                
                # Normalize pixel values to [0, 1]
                img_array = img_array / 255.0
                
                images.append(img_array)
                labels.append(class_idx)
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    print(f"Loaded {len(X)} images with shape {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    return X, y, class_names

def create_data_generators(X_train, y_train, X_val, y_val, batch_size=32):
    """
    Create data generators with augmentation for training
    """
    print("Creating data generators with augmentation...")
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # No augmentation for validation
    val_datagen = ImageDataGenerator()
    
    # Create generators
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
    
    return train_generator, val_generator

def visualize_samples(X, y, class_names, num_samples=8):
    """
    Display sample images from each class
    """
    print("Displaying sample images...")
    
    plt.figure(figsize=(12, 8))
    for i in range(num_samples):
        plt.subplot(2, 4, i + 1)
        plt.imshow(X[i])
        plt.title(f"{class_names[y[i]]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png')  # Save for your report
    plt.show()

def save_dataset_info(X_train, y_train, X_val, y_val, class_names):
    """
    Save dataset information for model training
    """
    print("Saving dataset information...")
    
    # Save the actual data arrays
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_val.npy', X_val)
    np.save('y_val.npy', y_val)
    np.save('class_names.npy', class_names)
    
    # Save dataset info
    dataset_info = {
        'input_shape': X_train.shape[1:],  # (224, 224, 3)
        'num_classes': len(class_names),
        'class_names': class_names,
        'train_samples': len(X_train),
        'val_samples': len(X_val)
    }
    
    import pickle
    with open('dataset_info.pkl', 'wb') as f:
        pickle.dump(dataset_info, f)
    
    print(f"Dataset info: {dataset_info}")
    print("Data saved successfully!")

def main():
    """
    Main function to run all data processing steps
    """
    # Update this path to where you extracted the dataset
    data_path = "RiceLeafs/rice_leaf_diseases"
    
    if not os.path.exists(data_path):
        print(f"Dataset path '{data_path}' not found!")
        print("Please download the dataset and update the data_path variable.")
        return
    
    # Step 1: Explore the dataset
    class_names = explore_dataset(data_path)
    
    # Step 2: Load and preprocess data
    X, y, class_names = load_and_preprocess_data(data_path)
    
    # Step 3: Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Data split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    
    # Step 4: Create data generators
    train_generator, val_generator = create_data_generators(X_train, y_train, X_val, y_val)
    
    # Step 5: Save dataset information (simpler approach)
    save_dataset_info(X_train, y_train, X_val, y_val, class_names)
    
    # Step 6: Visualize samples
    visualize_samples(X_train, y_train, class_names)
    
    print("Data processing completed successfully!")
    return train_generator, val_generator, class_names

if __name__ == "__main__":
    train_gen, val_gen, classes = main()