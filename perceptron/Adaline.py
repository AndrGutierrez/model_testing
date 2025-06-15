import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

class Adaline(Layer):
    def __init__(self, units=1):
        super(Adaline, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b  # Linear activation

# # Build the model
# model = Sequential([
#     Adaline(units=1)
# ])

# # Use Mean Squared Error (MSE) as loss (Adaline uses SSE/MSE)
# model.compile(optimizer=SGD(learning_rate=0.01), loss='mse')

# # Train the model (X_train, y_train should be numpy arrays)
# model.fit(X_train, y_train, epochs=50, batch_size=1)