import numpy as np

class NeuralNetwork:
    def __init__(self, layers, hidden_activation='sigmoid', output_activation='sigmoid'):
        self.layers = []
        layer_sizes = layers
        
        for i in range(1, len(layer_sizes)):
            self.layers.append(Linear(layer_sizes[i-1], layer_sizes[i]))
            if i < len(layer_sizes) - 1:
                self.layers.append(self.get_activation(hidden_activation))
            else:
                self.layers.append(self.get_activation(output_activation))
    
    def get_activation(self, activation):
        if activation == 'sigmoid':
            return Sigmoid()
        elif activation == 'relu':
            return ReLU()
        elif activation == 'softmax':
            return Softmax()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backward(self, loss):
        for layer in reversed(self.layers):
            loss = layer.backward(loss)
    
    def train(self, X, y, epochs, learning_rate, loss_function='mse'):
        for _ in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute loss
            if loss_function == 'mse':
                loss_obj = MSELoss()
                loss = loss_obj(output, y)
            elif loss_function == 'crossentropy':
                loss_obj = CrossEntropyLoss()
                loss = loss_obj(output, y)
            else:
                raise ValueError(f"Unsupported loss function: {loss_function}")
            
            # Backward pass
            self.backward(loss_obj.backward())
            
            # Update parameters
            for layer in self.layers:
                if isinstance(layer, Linear):
                    layer.weight -= learning_rate * layer.weight_grad
                    layer.bias -= learning_rate * layer.bias_grad
    
    def predict(self, X):
        return self.forward(X)

class Linear:
    def __init__(self, in_features, out_features):
        self.weight = np.random.randn(in_features, out_features) * 0.01
        self.bias = np.zeros((1, out_features))
        self.weight_grad = np.zeros_like(self.weight)
        self.bias_grad = np.zeros_like(self.bias)
    
    def __call__(self, x):
        self.input = x
        return np.dot(x, self.weight) + self.bias
    
    def backward(self, grad_output):
        self.weight_grad += np.dot(self.input.T, grad_output)
        self.bias_grad += np.sum(grad_output, axis=0, keepdims=True)
        return np.dot(grad_output, self.weight.T)

class Sigmoid:
    def __call__(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    
    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)

class ReLU:
    def __call__(self, x):
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        return grad_output * (self.input > 0)

class Softmax:
    def __call__(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output
    
    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)

class MSELoss:
    def __call__(self, y_pred, y_true):
        self.diff = y_pred - y_true
        return np.mean(self.diff**2)
    
    def backward(self):
        return 2 * self.diff / self.diff.size

class CrossEntropyLoss:
    def __call__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / y_true.shape[0]
    
    def backward(self):
        return (self.y_pred - self.y_true) / self.y_true.shape[0]

if __name__ == '__main__':
    nn = NeuralNetwork(layers=[2,4,1], hidden_activation='relu', output_activation='sigmoid')
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    nn.train(X, y, epochs=10000, learning_rate=0.01, loss_function='mse')
    predictions = nn.predict(X)
    print(predictions)