from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Reshape, Dropout
from sklearn.metrics import roc_curve, auc, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


def build_lstm_model(input_past_shape, input_future_shape, forecast_steps, num_targets):
    """
    Builds an LSTM model with two input branches.

    Parameters:
        input_past_shape (tuple): Shape of past input (time_steps, num_features).
        input_future_shape (tuple): Shape of future weather input (forecast_steps, num_weather_features).
        forecast_steps (int): Number of forecast time steps.
        num_targets (int): Number of target variables to predict.

    Returns:
        model: Compiled LSTM model.
    """
    # Branch 1: For X_train_past
    past_input = Input(shape=input_past_shape, name="past_input")
    x1 = LSTM(64, return_sequences=False)(past_input)

    # Branch 2: For X_train_weather_future
    future_input = Input(shape=input_future_shape, name="future_input")
    x2 = LSTM(32, return_sequences=False)(future_input)

    # Combine the two branches
    combined = Concatenate()([x1, x2])

    # Fully connected layers
    x = Dense(128, activation="relu")(combined)
    x = Dense(forecast_steps * num_targets, activation="linear")(x)

    # Reshape to match output shape: (batch_size, forecast_steps, num_targets)
    output = Reshape((forecast_steps, num_targets))(x)

    # Define the model
    model = Model(inputs=[past_input, future_input], outputs=output)
    model.compile(optimizer="adam", loss="mse")
    return model

def build_mc_dropout_lstm(input_past_shape, input_future_shape, forecast_steps, num_targets):
    """
    Builds an LSTM model with Monte Carlo Dropout for uncertainty estimation.

    Parameters:
        input_past_shape (tuple): Shape of past input (time_steps, num_features).
        input_future_shape (tuple): Shape of future weather input (forecast_steps, num_weather_features).
        forecast_steps (int): Number of forecast time steps.
        num_targets (int): Number of target variables to predict.

    Returns:
        model: Compiled LSTM model.
    """
    # Branch 1: For X_train_past
    past_input = Input(shape=input_past_shape, name="past_input")
    x1 = LSTM(64, return_sequences=False)(past_input)
    x1 = Dropout(0.2)(x1)

    # Branch 2: For X_train_weather_future
    future_input = Input(shape=input_future_shape, name="future_input")
    x2 = LSTM(32, return_sequences=False)(future_input)
    x2 = Dropout(0.2)(x2)

    # Combine the two branches
    combined = Concatenate()([x1, x2])

    # Fully connected layers
    x = Dense(128, activation="relu")(combined)
    x = Dropout(0.2)(x)  # MC Dropout layer
    x = Dense(forecast_steps * num_targets, activation="linear")(x)

    # Reshape to match output shape: (batch_size, forecast_steps, num_targets)
    output = Dense(forecast_steps * num_targets, activation="linear")(x)

    # Define the model
    model = Model(inputs=[past_input, future_input], outputs=output)
    model.compile(optimizer="adam", loss="mse")
    return model

def batch_predict(model, X_past, X_future, batch_size, num_passes):
    """
    Perform stochastic forward passes in batches to reduce memory usage.

    Parameters:
        model: Trained model with MC Dropout enabled.
        X_past: Past input data of shape (num_samples, time_steps, num_features).
        X_future: Future input data of shape (num_samples, forecast_steps, num_features).
        batch_size: Number of samples to process per batch.
        num_passes: Number of stochastic forward passes.

    Returns:
        all_predictions (numpy array): Predictions of shape 
                                        (num_passes, num_samples, forecast_steps, num_targets).
    """
    num_samples = X_past.shape[0]
    all_predictions = []

    for _ in range(num_passes):
        predictions = []
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_past = X_past[start_idx:end_idx]
            batch_future = X_future[start_idx:end_idx]
            pred = model.predict([batch_past, batch_future], verbose=0)
            predictions.append(pred)
        all_predictions.append(np.concatenate(predictions, axis=0))

    return np.array(all_predictions)
    

def plot_roc_curve(Y_test, predictions, target_index=0):
    """
    Plots the ROC curve for one of the target variables.

    Parameters:
        Y_test (numpy array): True values of shape (num_samples, forecast_steps, num_targets).
        predictions (numpy array): Predicted values of the same shape.
        target_index (int): Index of the target variable to plot the ROC for.

    Returns:
        None
    """
    true_binary = Y_test[:, :, target_index].ravel()  # Flatten for all forecast steps
    pred_probs = predictions[:, :, target_index].ravel()

    fpr, tpr, _ = roc_curve(true_binary, pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

def plot_residuals(Y_test, predictions, target_index, target_name):
    """
    Plots residuals (true - predicted) for a specific target variable.

    Parameters:
        Y_test (numpy array): True values of shape (num_samples, forecast_steps, num_targets).
        predictions (numpy array): Predicted values of the same shape.
        target_index (int): Index of the target variable to plot residuals for.
        target_name (str): Name of the target variable (e.g., 'grid', 'solar').

    Returns:
        None
    """
    true_values = Y_test[:, :, target_index].ravel()
    pred_values = predictions[:, :, target_index].ravel()
    residuals = true_values - pred_values

    plt.figure(figsize=(8, 6))
    plt.scatter(pred_values, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel(f'Predicted Values ({target_name})')
    plt.ylabel('Residuals (True - Predicted)')
    plt.title(f'Residual Plot for {target_name}')
    plt.grid()
    plt.show()
    
def plot_error_distribution(Y_test, predictions, target_index, target_name):
    """
    Plots the distribution of residuals (true - predicted) for a specific target variable.

    Parameters:
        Y_test (numpy array): True values of shape (num_samples, forecast_steps, num_targets).
        predictions (numpy array): Predicted values of the same shape.
        target_index (int): Index of the target variable to plot error distribution for.
        target_name (str): Name of the target variable (e.g., 'grid', 'solar').

    Returns:
        None
    """
    true_values = Y_test[:, :, target_index].ravel()
    pred_values = predictions[:, :, target_index].ravel()
    residuals = true_values - pred_values

    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel('Residuals (True - Predicted)')
    plt.ylabel('Frequency')
    plt.title(f'Error Distribution for {target_name}')
    plt.grid()
    plt.show()

def moving_average(data, window_size):
    """
    Computes the moving average of a 1D array.

    Parameters:
        data (array-like): Input data to smooth.
        window_size (int): Size of the moving average window.

    Returns:
        smoothed_data (np.ndarray): Smoothed data.
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    