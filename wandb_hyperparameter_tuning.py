import wandb
from wandb.integration.keras import WandbCallback
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from Module2_final_FeedforwardNN import FeedforwardNN

# Load dataset and normalize data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
X_train, X_val, X_test = X_train / 255.0, X_val / 255.0, X_test / 255.0

#sweep configs
sweep_config = {
    "method": "random", 
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "epochs": {"values": [5, 10]},
        "hidden_layers": {"values": [3, 4, 5]},
        "hidden_size": {"values": [32, 64, 128]},
        "weight_decay": {"values": [0, 0.0005, 0.5]},
        "learning_rate": {"values": [1e-3, 1e-4]},
        "optimizer": {"values": ["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"]},
        "batch_size": {"values": [16, 32, 64]},
        "weight_init": {"values": ["random", "xavier"]},
        "activation": {"values": ["sigmoid", "tanh", "relu"]}
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project="DL_A1_final_v2")

def train():
    wandb.init(
        project="DL_A1-final", 
        name=f"hl_{wandb.config.hidden_layers}_bs_{wandb.config.batch_size}_ac_{wandb.config.activation}"
    )
    config = wandb.config

    # Ensure hidden_layers is a list
    hidden_layers = [config.hidden_size] * config.hidden_layers

    # Initialize model
    model = FeedforwardNN(
        input_size=784,
        hidden_layers=hidden_layers,
        output_size=10,
        learning_rate=config.learning_rate,
        optimizer=config.optimizer
    )

    for epoch in range(config.epochs):
        # Train the model for one epoch
        model.train(
            X_train.reshape(-1, 784),
            np.eye(10)[y_train],
            epochs=1,  
            batch_size=config.batch_size
        )
        #loss and accuracy on validation set
        y_pred_val = model.forward(X_val.reshape(-1, 784))
        val_loss = -np.sum(np.eye(10)[y_val] * np.log(y_pred_val)) / X_val.shape[0]
        val_accuracy = np.mean(np.argmax(y_pred_val, axis=1) == y_val)

        # Compute loss and accuracy on training set
        y_pred_train = model.forward(X_train.reshape(-1, 784))
        train_loss = -np.sum(np.eye(10)[y_train] * np.log(y_pred_train)) / X_train.shape[0]
        train_accuracy = np.mean(np.argmax(y_pred_train, axis=1) == y_train)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "loss": train_loss,
            "accuracy": train_accuracy
        })

# Run sweep
wandb.agent(sweep_id, train, count=100)