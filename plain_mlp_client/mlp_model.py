import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.initializers import RandomUniform
import pandas as pd
import os
import sys
from positive_range_constraint import PositiveRangeConstraint
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learning_params import NUM_EPOCHS, BATCH_SIZE, CONSTRAINED
from seed_config import set_all_seeds

class MLPModel:
    """
    A simple MLP model for MNIST classification.
    This class provides methods to create, train and evaluate a plain model
    without federated learning or encryption.
    """

    def __init__(self, constraint=False):
        """Initialize the MLP model"""
        self.model = self._create_model()
        if constraint:
            self.model = self.create_constraint_model()

    def create_constraint_model(self):
        """
        Create and return a simple MLP model for MNIST with weight constraints.
        This model uses the PositiveRangeConstraint to enforce weight constraints
        on the Dense layers, ranging from the limited range of [0,1].
        """
        model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation='relu',
                  kernel_constraint=PositiveRangeConstraint(0.0, 1.0),
                  bias_constraint=PositiveRangeConstraint(0.0, 1.0),
                  kernel_initializer=RandomUniform(0.0, 1.0),
                  bias_initializer=RandomUniform(0.0, 1.0)),
            Dense(10, activation='softmax',
                  kernel_constraint=PositiveRangeConstraint(0.0, 1.0),
                  bias_constraint=PositiveRangeConstraint(0.0, 1.0),
                  kernel_initializer=RandomUniform(0.0, 1.0),
                  bias_initializer=RandomUniform(0.0, 1.0))
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    def _create_model(self):
        """Create and return a simple MLP model for MNIST"""
        model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, x_train, y_train, epochs=10, batch_size=64, validation_data=None):
        """
        Train the model on the provided data

        Args:
            x_train: Training features
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_data: Optional validation data tuple (x_val, y_val)

        Returns:
            History object containing training metrics
        """
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='accuracy', patience=5, min_delta=0.005, mode='max'
        )

        return self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=[early_stopping],
            verbose=1
        )

    def evaluate(self, x_test, y_test, verbose=0):
        """
        Evaluate the model on test data

        Args:
            x_test: Test features
            y_test: Test labels

        Returns:
            Tuple of (loss, accuracy)
        """
        return self.model.evaluate(x_test, y_test, verbose=verbose)

    def save_history_to_csv(self, history, csv_filepath):
        """Save training history to a CSV file"""
        if history is None:
            raise ValueError("No history found. Did you call .train() yet?")

        df = pd.DataFrame(history.history)
        df.to_csv(csv_filepath, index=False)
        print(f"Training history saved to {csv_filepath}")

    def save_model(self, filepath):
        """Save the model to a file"""
        # Save the model
        self.model.save(filepath)

    def load(self, filepath):
        """Load model from a file"""
        self.model = tf.keras.models.load_model(filepath)

def local_client_training_eval(constraint=False):
    """Train and evaluate a single MLP model on MNIST dataset"""
    set_all_seeds()
    # Load and preprocess the MNIST dataset
    dirpath = 'plain_mlp_model'
    csv_path = 'mlp_model_history.csv'
    model_path = "mnist_mlp.keras"
    if constraint:
        csv_path = 'constraint_mlp_model_history.csv'
        model_path = 'constraint_mlp_model.keras'
    model_epochs = NUM_EPOCHS
    model_batch_size = BATCH_SIZE
    os.makedirs(dirpath, exist_ok=True)
    print("Loading and preprocessing MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Create the model
    print(f"Creating MLP model with constraint: {'yes' if constraint else 'no'}...")
    mlp = MLPModel(constraint)
    # Train the model
    print("Training model...")
    history = mlp.train(
        x_train, y_train,
        epochs=model_epochs,
        batch_size=model_batch_size,
        validation_data=(x_test, y_test)
    )


    mlp.save_history_to_csv(history, f'{dirpath}/{csv_path}')
    # Evaluate the model
    print("\nEvaluating model...")
    loss, accuracy = mlp.evaluate(x_test, y_test)
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    print(f"Final Test Loss: {loss:.4f}")

    # Save the model
    mlp.save_model(f'{dirpath}/{model_path}')
    print(f'Model saved to {model_path}')


if __name__ == "__main__":
    local_client_training_eval(CONSTRAINED)