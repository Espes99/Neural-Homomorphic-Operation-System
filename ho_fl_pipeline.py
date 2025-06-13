from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import tensorflow as tf

from federated_learning_recorder import FederatedLearningRecorder
from learning_params import NUM_CLIENTS, NUM_ROUNDS, NUM_EPOCHS, BATCH_SIZE, CONSTRAINED, METHODS
from plain_mlp_client.mlp_model import MLPModel
from seed_config import set_all_seeds
from weights_util import encrypt_model_weights_ho, decrypt_model_weights_ho, add_encrypted_weights_ho, apply_masking_ho
import os
import numpy as np
import matplotlib.pyplot as plt

#Set global seed for reproducibility
set_all_seeds()
# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Keep original integer labels for dirichlet sampling
y_train_int = y_train.copy()
y_test_int = y_test.copy()

# y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Alice and Bob are defined in neural_crypto_ab_v2 as tf.keras.Model instances
HO_MODEL = os.path.join('models', 'HO_best_100.keras')
HO = tf.keras.models.load_model(HO_MODEL)


def generate_keys_for_model(model_weights):
    """Generate cryptographic keys for each layer of weights"""
    keys = []
    for w in model_weights:
        # Create one key per layer with appropriate batch size
        batch_size = w.flatten().shape[0]
        key = np.random.randint(0, 2, (batch_size, 32)).astype(np.float32)
        keys.append(key)
    return keys


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

# ### Draw client data distribution
# create a plot with the subplots of the clients and the number of label distributions
# fig, axs = plt.subplots(1, NUM_CLIENTS, figsize=(10, 5))
# for i in range(NUM_CLIENTS):
#     axs[i].hist(client_labels_int[i], bins=10, range=(0, 10), alpha=0.5)
#     axs[i].set_xlabel("Digit", fontsize=16)
#     axs[i].set_ylabel("Count", fontsize=16)
#     axs[i].tick_params(axis='both', which='major', labelsize=14)
#     axs[i].grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig(os.path.join('data-plots', 'client_data_distribution.png'))
# plt.show()

# Function to average weights from multiple models (HO - FedAvg)
def encrypted_sum(weights_list, model_keys):
    """Aggregate encrypted weights via HO additions."""
    total = weights_list[0]
    for w in weights_list[1:]:
        total = add_encrypted_weights_ho(total, w, HO, model_keys)
    return total


fl_recorder = FederatedLearningRecorder(num_clients=NUM_CLIENTS, method=METHODS[2], constraint=CONSTRAINED)

# Initialize the global model
global_model = MLPModel(constraint=CONSTRAINED)
global_weights_encrypted = None
global_weights_shapes = None
global_keys = generate_keys_for_model(global_model.model.get_weights())

# Store local models to preserve their weights for masking
local_models = [MLPModel(constraint=CONSTRAINED) for _ in range(NUM_CLIENTS)]
# Federated training parameters
num_rounds = NUM_ROUNDS  # number of communication rounds
local_epochs = NUM_EPOCHS  # epochs of training on each client per round
batch_size = BATCH_SIZE
curr_best_acc, best_round = 0, 0

# Federated training simulation
for round_num in range(num_rounds):
    print(f"\n--- Federated Training Round {round_num + 1} ---")
    local_weights = []

    # Store old local weights before training (for masking later)
    old_local_weights = []

    # Each client trains on its local data
    for client_index, (x_client, y_client) in enumerate(client_datasets):
        # Get the local model for this client
        local_model = local_models[client_index]

        # Save current weights before updating (for masking)
        old_weights = local_model.model.get_weights()
        old_local_weights.append(old_weights)

        if round_num == 0:
            # In round 0, use plaintext weights
            local_model.model.set_weights(global_model.model.get_weights())
        else:
            # In subsequent rounds, use masked weights (already set from previous round)
            pass  # Weights are already masked from previous round

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='accuracy', patience=5, min_delta=0.005, mode='max'
        )

        # Train the local model
        local_model.model.fit(x_client, y_client, epochs=local_epochs, batch_size=batch_size,
                              callbacks=[early_stopping], verbose=0)

        client_weights = local_model.model.get_weights()

        encrypted_weights, original_shapes = encrypt_model_weights_ho(client_weights, global_keys)
        local_weights.append(encrypted_weights)

        global_weights_shapes = original_shapes

        print(f"Client {client_index + 1} done.")

    # Average the weights from all clients (FedAvg)
    added_weights = encrypted_sum(local_weights, global_keys)
    decrypted_sum = decrypt_model_weights_ho(added_weights, global_weights_shapes, global_keys)
    avg_weights = [w_sum / NUM_CLIENTS for w_sum in decrypted_sum]

    # Apply masking to filter out noise from HO operations
    print(f"\n[MASKING] Round {round_num + 1}")
    masked_client_weights, round_rejected_stats = apply_masking_ho(
        old_local_weights,
        avg_weights,
        round_num,
        num_rounds,
    )
    fl_recorder.rejected_stats.append(round_rejected_stats)
    # Update each client's model with their masked weights
    for client_index in range(NUM_CLIENTS):
        local_models[client_index].model.set_weights(masked_client_weights[client_index])

    # For global model evaluation, encrypt the averaged weights
    global_weights_encrypted, global_weights_shapes = encrypt_model_weights_ho(avg_weights, global_keys)

    # For evaluation only: use the averaged weights (before masking) for global model
    global_model.model.set_weights(avg_weights)

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
print("\nFinal Test Accuracy (Global):", acc)
print(f"Final Best Accuracy: {curr_best_acc} in round {best_round}")
fl_recorder.save_fl_run_to_csv()
fl_recorder.save_rejected_to_csv()