import wandb
import argparse
import numpy as np
import torch
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import OneHotEncoder

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-wp", "--wandb_project", type=str, default="myprojectname", help="W&B project name")
parser.add_argument("-we", "--wandb_entity", type=str, default="myname", help="W&B entity (username or team name)")
args = parser.parse_args()

# Load dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize and reshape
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

# One-hot encode labels
encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))


# Initialize W&B
wandb.init(project=args.wandb_project, entity=args.wandb_entity)

class FeedforwardNN:
    def __init__(self, input_size, hidden_layers, output_size, 
                 learning_rate=0.1, optimizer="sgd", weight_init="random", 
                 weight_decay=0.0, loss="cross_entropy", batch_size=4, 
                 momentum=0.5, beta=0.5, beta1=0.5, beta2=0.5, epsilon=1e-6, 
                 activation="sigmoid"):
        """
        Neural network class supporting different optimizers, weight initializations, and loss functions.

        Parameters:
        - input_size: Number of input features
        - hidden_layers: List containing the number of neurons in each hidden layer
        - output_size: Number of output classes
        - learning_rate: Learning rate for optimization
        - optimizer: One of ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]
        - weight_init: "random" or "Xavier"
        - weight_decay: L2 regularization factor
        - loss: Loss function ("cross_entropy", "mean_squared_error")
        - batch_size: Batch size for training
        - momentum: Momentum factor (for momentum-based optimizers)
        - beta: Beta for RMSprop
        - beta1, beta2: Betas for Adam/Nadam
        - epsilon: Small value to avoid division by zero
        - activation: Activation function for hidden layers ("identity", "sigmoid", "tanh", "ReLU")
        """
        self.layers = [input_size] + hidden_layers + [output_size]
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.loss = loss
        self.batch_size = batch_size
        self.momentum = momentum
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.activation_func = activation
        self.t = 0  #Time step for Adam/Nadam
        
        # Initialize weights and biases
        self.weights = []
        for i in range(len(self.layers) - 1):
            if weight_init == "random":
                W = np.random.randn(self.layers[i], self.layers[i+1]) * 0.01
            elif weight_init == "Xavier":
                W = np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(1 / self.layers[i])
            self.weights.append(W)
        
        self.biases = [np.zeros((1, self.layers[i+1])) for i in range(len(self.layers) - 1)]

        # Optimizer-specific memory variables
        self.m_w = [np.zeros_like(w) for w in self.weights]  
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_w = [np.zeros_like(w) for w in self.weights]  
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.velocity_w = [np.zeros_like(w) for w in self.weights]
        self.velocity_b = [np.zeros_like(b) for b in self.biases]

    def activation(self, x):
        if self.activation_func == "identity":
            return x
        elif self.activation_func == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation_func == "tanh":
            return np.tanh(x)
        elif self.activation_func == "relu":
            return np.maximum(0, x)

    def activation_derivative(self, x):
        if self.activation_func == "identity":
            return np.ones_like(x)
        elif self.activation_func == "sigmoid":
            return x * (1 - x)
        elif self.activation_func == "tanh":
            return 1 - x ** 2
        elif self.activation_func == "relu":
            return (x > 0).astype(float)

    def loss_function(self, y_true, y_pred):
        if self.loss == "cross_entropy":
            y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
            return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]
        elif self.loss == "mean_squared_error":
            return np.mean((y_true - y_pred) ** 2)

    def forward(self, X):
        self.a = [X]
        for i in range(len(self.weights)):
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            self.a.append(self.activation(z))
        return self.a[-1]

    def backward(self, X, y):
        y_pred = self.a[-1]
        if self.loss == "cross_entropy":
            error = y_pred - y
        elif self.loss == "mean_squared_error":
            error = (y_pred - y) * self.activation_derivative(y_pred)
        
        grads_w = []
        grads_b = []
        for i in reversed(range(len(self.weights))):
            grads_w.insert(0, np.dot(self.a[i].T, error) / X.shape[0])
            grads_b.insert(0, np.sum(error, axis=0, keepdims=True) / X.shape[0])
            if i > 0:
                error = np.dot(error, self.weights[i].T) * self.activation_derivative(self.a[i])

        self.update_weights(grads_w, grads_b)

    def update_weights(self, grads_w, grads_b):
        self.t += 1  #time step for Adam/Nadam
        clip_value = 5  
        grads_w = [np.clip(g, -clip_value, clip_value) for g in grads_w]
        grads_b = [np.clip(g, -clip_value, clip_value) for g in grads_b]
        for i in range(len(self.weights)):
            grads_w[i] += self.weight_decay * self.weights[i]

            if self.optimizer == "sgd":
                self.weights[i] -= self.learning_rate * grads_w[i]
                self.biases[i] -= self.learning_rate * grads_b[i]

            elif self.optimizer == "momentum":
                self.velocity_w[i] = self.momentum * self.velocity_w[i] + (1 - self.momentum) * grads_w[i]
                self.velocity_b[i] = self.momentum * self.velocity_b[i] + (1 - self.momentum) * grads_b[i]
                self.weights[i] -= self.learning_rate * self.velocity_w[i]
                self.biases[i] -= self.learning_rate * self.velocity_b[i]

            elif self.optimizer == "nesterov":
                temp_w = self.weights[i] - self.momentum * self.velocity_w[i]
                temp_b = self.biases[i] - self.momentum * self.velocity_b[i]
                self.velocity_w[i] = self.momentum * self.velocity_w[i] + self.learning_rate * grads_w[i]
                self.velocity_b[i] = self.momentum * self.velocity_b[i] + self.learning_rate * grads_b[i]
                self.weights[i] = temp_w - self.learning_rate * self.velocity_w[i]
                self.biases[i] = temp_b - self.learning_rate * self.velocity_b[i]
            
            elif self.optimizer == "rmsprop":
                self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * np.square(grads_w[i])
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * np.square(grads_b[i])
                self.weights[i] -= self.learning_rate * grads_w[i] / (np.sqrt(self.v_w[i]) + self.epsilon)
                self.biases[i] -= self.learning_rate * grads_b[i] / (np.sqrt(self.v_b[i]) + self.epsilon)

            elif self.optimizer == "adam":
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grads_w[i]
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grads_b[i]
                self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * np.square(grads_w[i])
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * np.square(grads_b[i])

                m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
                m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
                v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
                v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

                self.weights[i] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
                self.biases[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

            elif self.optimizer == "nadam":
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grads_w[i]
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grads_b[i]
                self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * np.square(grads_w[i])
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * np.square(grads_b[i])

                m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
                m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
                v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
                v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

                # Nadam applies the Nesterov momentum term
                nesterov_term_w = (self.beta1 * m_w_hat) + ((1 - self.beta1) * grads_w[i]) / (1 - self.beta1 ** self.t)
                nesterov_term_b = (self.beta1 * m_b_hat) + ((1 - self.beta1) * grads_b[i]) / (1 - self.beta1 ** self.t)

                self.weights[i] -= self.learning_rate * nesterov_term_w / (np.sqrt(v_w_hat) + self.epsilon)
                self.biases[i] -= self.learning_rate * nesterov_term_b / (np.sqrt(v_b_hat) + self.epsilon)


    def train(self, X, y, epochs=1):
        for epoch in range(epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]

            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X[i:i+self.batch_size]
                y_batch = y[i:i+self.batch_size]
                self.forward(X_batch)
                self.backward(X_batch, y_batch)

            print(f"Epoch {epoch+1}, Loss: {self.loss_function(y, self.forward(X)):.4f}")

model = FeedforwardNN(
    input_size=784, 
    hidden_layers=[64] * 5,  # 5 hidden layers of size 64
    output_size=10, 
    learning_rate=1e-3, 
    optimizer="adam", 
    weight_init="Xavier",
    weight_decay=0,
    loss="cross_entropy", 
    batch_size=64, 
    activation="tanh"
)

# Train model
model.train(X_train, y_train, epochs=10)

# Compute Train Accuracy
y_train_pred = model.forward(X_train)  # Get model predictions
train_preds = np.argmax(y_train_pred, axis=1)
train_labels = np.argmax(y_train, axis=1)

train_correct = np.sum(train_preds == train_labels)
train_total = y_train.shape[0]
train_accuracy = train_correct / train_total

# Print and Log Train Accuracy
print(f"Train Accuracy: {train_accuracy:.4f}")
wandb.log({"Train Accuracy": train_accuracy})

wandb.finish()
