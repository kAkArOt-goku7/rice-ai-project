# train_model.py (updated plotting section)
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import pickle

def create_cnn_model(input_shape=(224, 224, 3), num_classes=3):
    """
    Create a Convolutional Neural Network from scratch
    """
    print("Building CNN Model from Scratch...")
    
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def load_training_data():
    """
    Load the preprocessed training data
    """
    print("Loading training data...")
    
    # Load dataset info
    with open('dataset_info.pkl', 'rb') as f:
        dataset_info = pickle.load(f)
    
    print(f"Dataset Info: {dataset_info}")
    
    # Load the actual data
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_val = np.load('X_val.npy')
    y_val = np.load('y_val.npy')
    
    print(f"Training data: {X_train.shape}, {y_train.shape}")
    print(f"Validation data: {X_val.shape}, {y_val.shape}")
    
    return X_train, y_train, X_val, y_val, dataset_info

def plot_training_history(history):
    """
    Improved plotting function with better formatting
    """
    # Set style for better appearance
    plt.style.use('default')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Accuracy
    ax1.plot(history.history['accuracy'], 'b-', linewidth=2, label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], 'r-', linewidth=2, label='Validation Accuracy')
    ax1.set_title('Model Accuracy During Training', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss
    ax2.plot(history.history['loss'], 'b-', linewidth=2, label='Training Loss')
    ax2.plot(history.history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    ax2.set_title('Model Loss During Training', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Improve layout
    plt.tight_layout()
    
    # Save with high quality
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main training function
    """
    print("Starting Model Training Pipeline...")
    print("=" * 50)
    
    # Load data
    X_train, y_train, X_val, y_val, dataset_info = load_training_data()
    
    # Create model
    model = create_cnn_model(dataset_info['input_shape'], dataset_info['num_classes'])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model Compiled Successfully!")
    model.summary()
    
    # Train the model
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        epochs=25,
        validation_data=(X_val, y_val),
        batch_size=32,
        verbose=1
    )
    
    # Save the trained model
    model.save('rice_disease_cnn.h5')
    print("\nModel saved as 'rice_disease_cnn.h5'")
    
    # Plot training history
    plot_training_history(history)
    
    # Print final results
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Classes: {dataset_info['class_names']}")
    print("=" * 50)

if __name__ == "__main__":
    main()