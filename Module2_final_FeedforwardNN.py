import numpy as np

class FeedforwardNN:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01, optimizer="sgd", weight_init="random", weight_decay=0):
        """
        input_size: Number of input features
        hidden_layers: List containing the number of neurons in each hidden layer
        output_size: Number of output classes
        learning_rate: Learning rate for optimization
        optimizer: One of ["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"]
        weight_init: "random" or "xavier"
        weight_decay: L2 regularization factor
        """
        self.layers = [input_size] + hidden_layers + [output_size]
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.weight_decay = weight_decay  # weight decay factor
        self.beta1 = 0.9  
        self.beta2 = 0.999  
        self.epsilon = 1e-8  
        self.t = 0  

        #weights and biases
        self.weights = []
        for i in range(len(self.layers) - 1):
            if weight_init == "random":
                W = np.random.randn(self.layers[i], self.layers[i+1]) * 0.01
            elif weight_init == "xavier":
                W = np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(1 / self.layers[i])
            self.weights.append(W)
        
        self.biases = [np.zeros((1, self.layers[i+1])) for i in range(len(self.layers) - 1)]

        #momentum and history terms
        self.m_w = [np.zeros_like(w) for w in self.weights]  
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_w = [np.zeros_like(w) for w in self.weights]  
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.velocity_w = [np.zeros_like(w) for w in self.weights]
        self.velocity_b = [np.zeros_like(b) for b in self.biases]

    def update_weights(self, grads_w, grads_b):
        self.t += 1  # time step for Adam/Nadam

        for i in range(len(self.weights)):
            #L2 weight decay
            grads_w[i] += self.weight_decay * self.weights[i]

            if self.optimizer == "sgd":
                self.weights[i] -= self.learning_rate * grads_w[i]
                self.biases[i] -= self.learning_rate * grads_b[i]

            elif self.optimizer == "momentum":
                self.velocity_w[i] = self.beta1 * self.velocity_w[i] + (1 - self.beta1) * grads_w[i]
                self.velocity_b[i] = self.beta1 * self.velocity_b[i] + (1 - self.beta1) * grads_b[i]
                self.weights[i] -= self.learning_rate * self.velocity_w[i]
                self.biases[i] -= self.learning_rate * self.velocity_b[i]

            elif self.optimizer == "nesterov":
                temp_w = self.weights[i] - self.beta1 * self.velocity_w[i]
                temp_b = self.biases[i] - self.beta1 * self.velocity_b[i]
                self.velocity_w[i] = self.beta1 * self.velocity_w[i] + self.learning_rate * grads_w[i]
                self.velocity_b[i] = self.beta1 * self.velocity_b[i] + self.learning_rate * grads_b[i]
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
    def train(self, X, y, epochs=10, batch_size=32):
        """
        Train the neural network
        """
        for epoch in range(epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                self.forward(X_batch)
                self.backward(X_batch, y_batch)

            if epoch % 2 == 0:
                loss = -np.sum(y * np.log(self.forward(X))) / X.shape[0]
                print(f"Epoch {epoch}, Loss: {loss:.4f}")