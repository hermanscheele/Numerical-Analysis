import numpy as np


def single_layer_prediction(x, W, b, activation='sigmoid'):
    """
    Performs a single forward pass prediction with a single-layer neural network.

    Parameters:
    -----------
    x : numpy.ndarray
        Input vector of shape (n_features,) or (n_features, n_samples)
    W : numpy.ndarray
        Weight matrix of shape (n_outputs, n_features)
    b : numpy.ndarray
        Bias vector of shape (n_outputs,)
    activation : str, optional
        Activation function to use ('sigmoid', 'relu', 'tanh', or 'linear')
        Default is 'sigmoid'

    Returns:
    --------
    numpy.ndarray
        Output predictions of shape (n_outputs,) or (n_outputs, n_samples)

    Example:
    --------
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> W = np.array([[0.5, -0.2, 0.1], [0.3, 0.4, -0.1]])
    >>> b = np.array([0.1, -0.2])
    >>> output = single_layer_prediction(x, W, b)
    """
    z = np.dot(W, x) + b

    if activation == 'sigmoid':
        return 1 / (1 + np.exp(-z))
    elif activation == 'relu':
        return np.maximum(0, z)
    elif activation == 'tanh':
        return np.tanh(z)
    elif activation == 'linear':
        return z
    else:
        raise ValueError(f"Unknown activation function: {activation}")


if __name__ == "__main__":
    # Example usage
    print("Single Layer Neural Network Prediction Example\n")

    # Define input
    x = np.array([1.0, 2.0, 3.0])
    print(f"Input: {x}")

    # Define weights and bias for a layer with 2 output neurons
    W = np.array([[0.5, -0.2, 0.1],
                  [0.3, 0.4, -0.1]])
    b = np.array([0.1, -0.2])

    print(f"\nWeights shape: {W.shape}")
    print(f"Bias shape: {b.shape}")

    # Make prediction with different activation functions
    output_sigmoid = single_layer_prediction(x, W, b, activation='sigmoid')
    output_relu = single_layer_prediction(x, W, b, activation='relu')
    output_tanh = single_layer_prediction(x, W, b, activation='tanh')

    print(f"\nOutput (sigmoid): {output_sigmoid}")
    print(f"Output (relu): {output_relu}")
    print(f"Output (tanh): {output_tanh}")
