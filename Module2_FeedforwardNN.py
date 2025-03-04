import numpy as np

class FeedforwardNN:
    def __init__(self, input_size, hidden_layers, output_size):
        """
        input_size: Number of input features (e.g., 784 for 28x28 images)
        hidden_layers: List containing the number of neurons in each hidden layer
        output_size: Number of output classes (10 for Fashion-MNIST)
        """
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = [np.random.randn(self.layers[i], self.layers[i + 1]) * 0.01 
                        for i in range(len(self.layers) - 1)] #weights is a matrix, containing matrices of weight from layer l to layer l+1
        self.biases = [np.zeros((1, self.layers[i + 1])) for i in range(len(self.layers) - 1)]

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  #numerical stability
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        """
        X: Input matrix of shape (batch_size, input_size)
        Returns: Output probability distribution of shape (batch_size, output_size)
        """
        activations = X
        for i in range(len(self.weights) - 1):
            activations = self.relu(np.dot(activations, self.weights[i]) + self.biases[i])
        output = self.softmax(np.dot(activations, self.weights[-1]) + self.biases[-1])
        return output

nn = FeedforwardNN(input_size=784, hidden_layers=[128, 64], output_size=10)

X_sample = np.random.randn(5, 784)  #for testing we pass 5 images, each image is of shape(1,784)
output = nn.forward(X_sample)
print(output)  #we get 5 probability distributions for each input