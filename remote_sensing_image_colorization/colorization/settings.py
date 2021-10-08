from tensorflow.keras.optimizers import Adam

LEARNING_RATE = 0.01
LOSS = 'mean_squared_error'
HEIGHT = WIDTH = 224
DEFAULT_OPTIMIZER = Adam(learning_rate=LEARNING_RATE)