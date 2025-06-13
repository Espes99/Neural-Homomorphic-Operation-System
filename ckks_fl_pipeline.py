from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import os
import sys
from encryption import create_ckks_context
from federated_learning_recorder import FederatedLearningRecorder
from learning_params import NUM_CLIENTS, NUM_ROUNDS, NUM_EPOCHS, BATCH_SIZE, METHODS, CONSTRAINED
from seed_config import set_all_seeds
from weights_util import encrypt_model_weights, decrypt_model_weights
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plain_mlp_client.mlp_model import MLPModel

#Set global seed for reproducibility
set_all_seeds()

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

#Keep original integer labels for dirichlet sampling
y_train_int = y_train.copy()
y_test_int = y_test.copy()

# y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Simulate federated clients by splitting the training data
# HO Dirichlet concentration
alpha = 1.0
digit_indices = {i: np.where(y_train_int == i)[0] for i in range(10)}
client_indices = {i: [] for i in range(NUM_CLIENTS)}

for digit, idxs in digit_indices.items():
    # shuffle
    np.random.shuffle(idxs)
    # sample proportions for each client
    proportions = np.random.dirichlet(alpha=np.repeat(alpha, NUM_CLIENTS))
    # compute split sizes
    counts = (proportions * len(idxs)).astype(int)
    # adjust to ensure sum equals
    counts[-1] = len(idxs) - sum(counts[:-1])
    start = 0
    for client_id, count in enumerate(counts):
        client_indices[client_id].extend(idxs[start:start + count])
        start += count

client_datasets = []
client_labels_int = []  # Keep for plotting
for i in range(NUM_CLIENTS):
    idx = client_indices[i]
    x_client = x_train[idx]
    y_client_int = y_train_int[idx]
    y_client_onehot = to_categorical(y_client_int, 10)

    client_datasets.append((x_client, y_client_onehot))
    client_labels_int.append(y_client_int)

# Function to average weights from multiple models (HO - FedAvg)
def fed_avg(weights_list):
    avg_weights = []
    number_of_clients = len(weights_list)

    # Iterate through each layer's encrypted weights
    for layer_idx in range(len(weights_list[0])):
        # Get the encrypted weights for this layer from all clients
        layer_weights = [weights[layer_idx] for weights in weights_list]

        # Start with the first client's weights for this layer
        sum_weight = layer_weights[0]
        # someweight = someweight + layer_weights[i]
        # pyseal
        # Homomorphic addition of weights from other clients
        for index in range(1, number_of_clients):
            sum_weight = sum_weight + layer_weights[index]

        # Scale by 1/number_of_clients to get average using homomorphic multiplication
        avg_weight = sum_weight * (1.0 / number_of_clients)

        avg_weights.append(avg_weight)

    return avg_weights


fl_recorder = FederatedLearningRecorder(num_clients=NUM_CLIENTS, method=METHODS[1], constraint=CONSTRAINED)

# Initialize the global model
global_model = MLPModel(constraint=CONSTRAINED)
global_weights_encrypted = None
global_weights_shapes = None
ckks_context = create_ckks_context()
# Federated training parameters
num_rounds = NUM_ROUNDS  # number of communication rounds
local_epochs = NUM_EPOCHS  # epochs of training on each client per round
batch_size = BATCH_SIZE
curr_best_acc, best_round = 0, 0
# Federated training simulation
for round_num in range(num_rounds):
    print(f"\n--- Federated Training Round {round_num + 1} ---")
    local_weights = []

    # Each client trains on its local data
    for client_index, (x_client, y_client) in enumerate(client_datasets):
        # Create a new local model
        local_model = MLPModel(constraint=CONSTRAINED)

        if round_num == 0:
            # In round 0, use plaintext weights
            local_model.model.set_weights(global_model.model.get_weights())
        else:
            decrypted_weights = decrypt_model_weights(global_weights_encrypted, global_weights_shapes, ckks_context)
            local_model.model.set_weights(decrypted_weights)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='accuracy', patience=5, min_delta=0.005, mode='max'
        )

        # Train the local model
        local_model.model.fit(x_client, y_client, epochs=local_epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=0)

        client_weights = local_model.model.get_weights()
        encrypted_weights, original_shapes = encrypt_model_weights(client_weights, ckks_context)
        local_weights.append(encrypted_weights)

        global_weights_shapes = original_shapes

        print(f"Client {client_index + 1} done.")

    # Average the weights from all clients (FedAvg)
    global_weights_encrypted = fed_avg(local_weights)

    # For evaluation only: decrypt weights to update the global model
    decrypted_global_weights = decrypt_model_weights(global_weights_encrypted, global_weights_shapes, ckks_context)
    global_model.model.set_weights(decrypted_global_weights)

    # evaluation client

    # Evaluate the global model on the test data after each round
    loss, acc = global_model.evaluate(x_test, y_test, verbose=0)
    fl_recorder.add_round_metrics(round_num=round_num+1, loss=loss, accuracy=acc)
    if acc > curr_best_acc:
        curr_best_acc, best_round = acc, round_num + 1
        print(f"New Best Accuracy: {curr_best_acc:.4f} at round {best_round}")
    print(f"========= Round {round_num + 1} ==========")
    print(f"--------- Accuracy: {acc:.4f} ---------")
    print(f"--------- LOSS {loss} ---------")

# Final evaluation of the global model
loss, acc = global_model.evaluate(x_test, y_test, verbose=0)
print("\nFinal Test Accuracy:", acc)
print(f"Final Best Accuracy: {curr_best_acc} in round {best_round}")
fl_recorder.save_fl_run_to_csv()