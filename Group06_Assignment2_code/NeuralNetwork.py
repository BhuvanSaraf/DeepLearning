import numpy as np

def logistic(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def logistic_derivative(a):
    return a * (1 - a)


def tanh_derivative(a):
    return 1 - a**2


def relu_derivative(a):
    return (a > 0).astype(float)


ACTIVATIONS = {
    "logistic": logistic,
    "tanh": tanh,
    "relu": relu
}


ACTIVATION_DERIVATIVES = {
    "logistic": logistic_derivative,
    "tanh": tanh_derivative,
    "relu": relu_derivative,
    "linear": lambda x: np.ones_like(x)
}

class NeuralNetwork:

    def __init__(
        self,
        input_size,
        net_dims,
        activation = "tanh",
        type = 'classification',
        seed=None,
        learning_rate=0.02,
        weight_decay = 0.001,
        batch_size = 1,
        epochs = 100
    ):
        self.input_size = input_size
        self.net_dims = net_dims
        self.activation = activation
        self.type = type
        self.seed = seed
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        # Initialize weights
        self.W , self.b = self._init()
        # If regression, enforce single-output final layer
        if self.type == 'regression' and self.net_dims[-1] != 1:
            raise ValueError("For regression, net_dims[-1] must be 1 (single linear output)")
        self.error_history_ = []
        # These will be populated during forward()
        self.A = []
        self.Z = []

        
    def _init(self):

        net_dims = self.net_dims
        activation = self.activation
        seed = self.seed
        
        self.rng = np.random.default_rng(seed)

        Ws = []
        bs = []

        n = len(net_dims) - 1
        
        for i in range(n):

            fan_in = self.net_dims[i]
            fan_out = self.net_dims[i + 1]
        
            activation_name = self.activation
        
            if activation_name == "relu":
                # He initialization
                std = np.sqrt(2.0 / fan_in)
        
            elif activation_name == "logistic":
                std = np.sqrt(1.0 / fan_in)

            elif activation_name in ("tanh", "softmax", "linear"):
                std = np.sqrt(2.0 / (fan_in + fan_out))

            else:
                raise ValueError(f"Unknown activation: {activation_name}")
        
            w = self.rng.standard_normal((fan_in, fan_out)) * std
            b = np.zeros(fan_out)
        
            Ws.append(w)
            bs.append(b)

        return Ws, bs

    def predict(self, x):

            a = x
            # Hidden layers
            activation = ACTIVATIONS[self.activation]
            for i in range(len(self.W) - 1):
    
                # Linear transformation
                z = np.dot(a, self.W[i]) + self.b[i]
                a = activation(z)
    
            # Output layer

            z = np.dot(a, self.W[-1]) + self.b[-1]

            if self.type == 'classification':
                # squared-error training uses the same bounded activation
                # at the output as in the hidden layers (matches the
                # single-neuron logistic/tanh perceptron from Assignment 1,
                # extended to a one-hot output vector)
                y_pred = activation(z)
            else:  # regression: linear output (single neuron)
                # ensure output is 2D (n_samples, 1)
                y_pred = z if z.ndim == 2 else z.reshape(-1, 1)
                
            # For regression return a 1-D array for convenience
            if self.type == 'regression':
                return y_pred.reshape(-1)

            return y_pred
    
    
    
    def forward(self, x):

        self.A = [x]
        self.Z = []

        a = x

        # Hidden layers
        activation = ACTIVATIONS[self.activation]
        for i in range(len(self.W) - 1):

            # Linear transformation
            z = np.dot(a, self.W[i]) + self.b[i]
            self.Z.append(z)
            a = activation(z)
            self.A.append(a)

        # Output layer

        z = np.dot(a, self.W[-1]) + self.b[-1]

        self.Z.append(z)

        if self.type == 'classification':
            # same activation as hidden layers, applied elementwise to
            # each output node -- NOT softmax. This is required so that
            # backprop with squared error loss (below) is valid.
            y_pred = activation(z)
        else:  # regression: linear output (single neuron)
            # ensure output is 2D (n_samples, 1)
            y_pred = z if z.ndim == 2 else z.reshape(-1, 1)

        self.A.append(y_pred)

        return y_pred


    def backward( self, y):

        # Ensure y has the correct shape for regression
        if self.type == 'regression' and y.ndim == 1:
            y = y.reshape(-1, 1)

        batch_size = y.shape[0]

        # Gradient storage
        dW = [None] * len(self.W)
        db = [None] * len(self.b)

        # Activation derivative (same function used for hidden AND,
        # for classification, the output layer)
        derivative = ACTIVATION_DERIVATIVES[self.activation]

        # OUTPUT LAYER
        # squared error loss: E = 1/2 * sum((y - y_pred)^2)
        # for regression the output activation is linear (derivative = 1)
        # for classification the output activation matches the hidden
        # activation, so its derivative must be applied here too
        error = self.A[-1] - y
        if self.type == 'classification':
            delta = error * derivative(self.A[-1])
        else:
            delta = error  # linear output, derivative is 1

        # Weight gradient
        dW[-1] = (self.A[-2].T @ delta) / batch_size

        # Bias gradient
        db[-1] = np.sum(delta, axis=0) / batch_size

        # HIDDEN LAYERS
        for i in range(len(self.W) - 2, -1, -1):            

            # Propagate error backwards
            delta = delta @ self.W[i + 1].T

            # Apply activation derivative
            delta *= derivative(self.A[i+1])

            # Weight gradient
            dW[i] = (self.A[i].T @ delta) / batch_size

            # Bias gradient
            db[i] = np.sum(delta, axis=0) / batch_size

        return dW, db


    def fit(self, X, y):

        self.error_history_ = []
#-----------------------------------------------------------
        # Validate batch_size
        batch_size = self.batch_size

        if batch_size < 0:
            raise ValueError("batch_size must be 0, 1, or a positive integer.")
        
        if batch_size > len(X):
            raise ValueError("batch_size cannot be greater than the number of samples.")
        
        if batch_size == 0:
            batch_size = len(X)

#------------------------------------------------------------
        #Training 
        for epoch in range(self.epochs):
             
            indices = self.rng.permutation(len(X))

            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for start in range(0, len(X), batch_size):
            
                end = start + batch_size
            
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
            
                self.forward(X_batch)
            
                dW, db = self.backward(y_batch)
            
                self.update(dW, db)

#-------------------------------------------------------------
            # to track the network
            y_pred = self.forward(X)

            # Compute loss depending on task -- BOTH use squared error,
            # per the assignment spec ("use squared error as
            # instantaneous loss function")
            if self.type == 'classification':
                train_loss = np.mean(np.sum((y_pred - y) ** 2, axis=1)) / 2
            else:  # regression: mean squared error
                yy = y
                if yy.ndim == 1:
                    yy = yy.reshape(-1, 1)
                train_loss = np.mean((y_pred - yy) ** 2) / 2

            self.error_history_.append(train_loss)

            
            #print(f"{train_loss:.7f} | {val_loss:.7f} || {train_acc:.7f} | {val_acc:.7f} |")

        return self

    def update(self, dW, db):

        for i in range(len(self.W)):

            # Update weights
            self.W[i] -= (self.learning_rate * dW[i] + self.learning_rate * self.W[i] * self.weight_decay)

            # Update biases
            self.b[i] -= (self.learning_rate * db[i] + self.learning_rate * self.b[i] * self.weight_decay)

 
if __name__ == '__main__':
    np.random.seed(1)
    X0 = np.random.normal(loc=[-2, -2], scale=0.5, size=(50, 2))
    X1 = np.random.normal(loc=[2, 2], scale=0.5, size=(50, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)

    nn = NeuralNetwork(
        input_size=2,
        net_dims=[2, 10, 1],
        activation='tanh',
        seed=1,
        learning_rate=0.01,
        batch_size=0,
    )
