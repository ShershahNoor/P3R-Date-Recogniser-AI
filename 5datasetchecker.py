import numpy as np
import matplotlib.pyplot as plt

def load_and_verify(filename="image_dataset.npz"):
    data = np.load(filename)
    images = data['images']
    
    print(f"Dataset Shape: {images.shape}") 
    
    images_normalized = images.astype('float32') / 255.0
    
    plt.imshow(images_normalized[0])
    plt.title("Sample from Database")
    plt.axis('off')
    plt.show()
    
    return images_normalized

if __name__ == "__main__":
    X_train = load_and_verify()