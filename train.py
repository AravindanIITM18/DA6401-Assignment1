import wandb
import argparse
import numpy as np
from tensorflow.keras.datasets import fashion_mnist, mnist
from sklearn.model_selection import train_test_split
from Module2_final_FeedforwardNN import FeedforwardNN

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train Feedforward Neural Network with WandB logging")

parser.add_argument("-wp", "--wandb_project", type=str, default="myprojectname", help="WandB project name")
parser.add_argument("-we", "--wandb_entity", type=str, default="myname", help="WandB entity name")
parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist", help="Dataset to use")
parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy", help="Loss function")
parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="adam", help="Optimizer")
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, help="Learning rate")
parser.add_argument("-m", "--momentum", type=float, default=0.5, help="Momentum for optimizers")
parser.add_argument("-beta", "--beta", type=float, default=0.5, help="Beta for RMSprop")
parser.add_argument("-beta1", "--beta1", type=float, default=0.5, help="Beta1 for Adam/Nadam")
parser.add_argument("-beta2", "--beta2", type=float, default=0.5, help="Beta2 for Adam/Nadam")
parser.add_argument("-eps", "--epsilon", type=float, default=0.000001, help="Epsilon for optimizers")
parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0, help="Weight decay for optimizers")
parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="Xavier", help="Weight initialization method")
parser.add_argument("-nhl", "--num_layers", type=int, default=5, help="Number of hidden layers")
parser.add_argument("-sz", "--hidden_size", type=int, default=64, help="Number of neurons per hidden layer")
parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="tanh", help="Activation function")

args = parser.parse_args()

# Initialize WandB
wandb.init(project=args.wandb_project, entity=args.wandb_entity)
wandb.run.name = f"hl_{args.num_layers}_bs_{args.batch_size}_ac_{args.activation}"
wandb.run.save()

# Load dataset
if args.dataset == "mnist":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
else:
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Preprocessing: Normalize images
X_train, X_val, X_test = X_train / 255.0, X_val / 255.0, X_test / 255.0

# Initialize model
hidden_layers = [args.hidden_size] * args.num_layers
model = FeedforwardNN(
    input_size=784,
    hidden_layers=hidden_layers,
    output_size=10,
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    optimizer=args.optimizer,
    batch_size=args.batch_size,
    weight_init=args.weight_init,
    activation=args.activation
)

# Training loop
for epoch in range(args.epochs):
    model.train(X_train.reshape(-1, 784), np.eye(10)[y_train], epochs=1)

    # Compute validation loss and accuracy
    y_pred_val = model.forward(X_val.reshape(-1, 784))
    y_pred_val = np.clip(y_pred_val, 1e-9, 1 - 1e-9)
    val_loss = -np.sum(np.eye(10)[y_val] * np.log(y_pred_val)) / X_val.shape[0]
    val_accuracy = np.mean(np.argmax(y_pred_val, axis=1) == y_val)

    # Compute training loss and accuracy
    y_pred_train = model.forward(X_train.reshape(-1, 784))
    y_pred_train = np.clip(y_pred_train, 1e-9, 1 - 1e-9)
    train_loss = -np.sum(np.eye(10)[y_train] * np.log(y_pred_train)) / X_train.shape[0]
    train_accuracy = np.mean(np.argmax(y_pred_train, axis=1) == y_train)

    # Log metrics to WandB
    wandb.log({
        "epoch": epoch + 1,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "loss": train_loss,
        "accuracy": train_accuracy
    })

wandb.finish()
