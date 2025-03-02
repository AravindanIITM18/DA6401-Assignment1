import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import os

os.environ["KERAS_HOME"] = os.getcwd()
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()#load dataset

print(f"Dataset stored in: {os.path.join(os.getcwd(), 'datasets')}")
#class labels
class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
#one sample per class
samples = []
for i in range(10):
    idx = np.where(y_train == i)[0][0]  #first occurrence of each class
    samples.append((x_train[idx], class_labels[i]))

#plot the samples in a grid
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, (image, label) in enumerate(samples):
    ax = axes[i // 5, i % 5]
    ax.imshow(image, cmap='gray')
    ax.set_title(label)
    ax.axis('off')

plt.tight_layout()
plt.show()