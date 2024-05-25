import torch
from torch import nn

def create_linear_regression_model(input_size, output_size):
    """
    Create a linear regression model with the given input and output sizes.
    """
    model = nn.Linear(input_size, output_size)
    return model

def train_iteration(X, y, model, loss_fn, optimizer):
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def fit_regression_model(X, y):
    """
    Train the model for the given number of epochs.
    """
    learning_rate = 0.001  # Adjusted learning rate for convergence
    num_epochs = 5000  # Increased number of epochs for better training
    input_features = X.shape[1]  # Number of input features
    output_features = y.shape[1] if len(y.shape) > 1 else 1  # Number of output features
    
    model = create_linear_regression_model(input_features, output_features)
    
    loss_fn = nn.MSELoss()  # Using Mean Squared Error Loss

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    previos_loss = float("inf")
    loss_change_threshold = 0.001  # Threshold for significant loss change

    for epoch in range(1, num_epochs + 1):
        loss = train_iteration(X, y, model, loss_fn, optimizer)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
        if abs(previos_loss - loss.item()) < loss_change_threshold:
            print(f"Early stopping at epoch {epoch}, Loss: {loss.item()}")
            break
        previos_loss = loss.item()

    return model, loss
