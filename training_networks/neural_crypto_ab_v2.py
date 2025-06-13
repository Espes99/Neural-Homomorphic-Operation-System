import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learning_params import SCALE
os.makedirs('training_networks/models', exist_ok=True)
os.makedirs('training_networks/results', exist_ok=True)
# -----------------------------
# Configuration
# -----------------------------
N_BITS     = 32       # 32 bits to represent one float (float32)
BATCH_SIZE = 512
N_ROUNDS   = 500000000     # Training iterations
PATIENCE   = 100000        # Early stopping patience

# -----------------------------
# Helper functions: float <-> bits
# -----------------------------

def float_to_bits(x):
    """
    Convert scalar or 1D array of floats into bit‐vectors of
    their scaled‐to‐int representation.
    """
    x_arr = np.atleast_1d(np.array(x, dtype=np.float32))
    # Round to nearest scaled integer
    x_int = np.round(x_arr * SCALE).astype(np.int32)             # shape (n,)
    # Extract bits: (n,32), big‐endian bit order
    bits = ((x_int[:, None] >> np.arange(31, -1, -1)) & 1).astype(np.uint8)
    return bits.astype(np.float32)  # if you need floats

def bits_to_float(bits):
    """
    Convert bit‐vectors back into floats by reconstructing the
    scaled integer and then dividing by SCALE.
    """
    # bits: (n,32) array of 0/1
    # Compute integer = sum bits * 2**(31-i)
    weights = (2 ** np.arange(31, -1, -1)).astype(np.int32)      # shape (32,)
    ints    = bits.astype(np.int32) @ weights                   # shape (n,)
    return ints.astype(np.float32) / SCALE

def sample_data(batch_size, n_bits):
    raw_floats    = np.random.rand(batch_size).astype(np.float32)
    messages_bits = float_to_bits(raw_floats)
    keys          = np.random.randint(0, 2, size=(batch_size, n_bits)).astype(np.float32)
    return messages_bits, keys, raw_floats

# -----------------------------
# Model definitions (Alice pads two zeros internally)
# -----------------------------
def make_alice_old():
    inp   = tf.keras.Input(shape=(N_BITS*2,))
    x     = tf.keras.layers.Dense(64, activation='relu')(inp)
    x     = tf.keras.layers.Dense(64, activation='relu')(x)
    tail  = tf.keras.layers.Dense(N_BITS - 2, activation='tanh')(x)
    zeros = tf.keras.layers.Lambda(lambda z: tf.zeros((tf.shape(z)[0], 2)), output_shape=(2,))(tail)
    cipher= tf.keras.layers.Concatenate(axis=1)([zeros, tail])
    return tf.keras.Model(inputs=inp, outputs=cipher)

def make_alice():
    """
    Alice model that outputs an unconstrained 32-dimensional ciphertext.
    """
    inp = tf.keras.Input(shape=(N_BITS * 2,))
    x   = tf.keras.layers.Dense(64, activation='relu')(inp)
    x   = tf.keras.layers.Dense(64, activation='relu')(x)
    # Directly emit all 32 output bits
    cipher = tf.keras.layers.Dense(N_BITS, activation='tanh')(x)
    return tf.keras.Model(inputs=inp, outputs=cipher)


Alice = make_alice()


def make_bob():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(N_BITS*2,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(N_BITS, activation='sigmoid')
    ])

Bob = make_bob()


def make_eve():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(N_BITS,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(N_BITS, activation='sigmoid')
    ])

Eve = make_eve()

# -----------------------------
# Optimizers & Loss functions
# -----------------------------
opt_AB = tf.keras.optimizers.legacy.RMSprop(learning_rate=1e-3)
opt_E  = tf.keras.optimizers.legacy.RMSprop(learning_rate=1e-3)

bce = tf.keras.losses.BinaryCrossentropy()

def bob_loss_fn(m, m_hat): return bce(m, m_hat)

def eve_loss_fn(m, m_eve): return bce(m, m_eve)

def ab_loss_fn(m, m_bob, m_eve):
    bob_err = bob_loss_fn(m, m_bob)
    eve_err = eve_loss_fn(m, m_eve)
    penalty = tf.nn.relu(0.693 - eve_err)
    return bob_err + penalty**2

# -----------------------------
# Training Loop
# -----------------------------
loss_history_ab  = []
loss_history_bob = []
loss_history_eve = []

best_loss = np.inf
best_loss_iteration = 0

for step in tqdm(range(N_ROUNDS), desc='Training AB+EVE'):
    m_bits, keys, _ = sample_data(BATCH_SIZE, N_BITS)

    # --- Train Alice & Bob ---
    with tf.GradientTape() as tape:
        mk_input = tf.concat([m_bits, keys], axis=1)
        c        = Alice(mk_input)         # first two bits always zero
        ck_input = tf.concat([c, keys], axis=1)
        m_bob    = Bob(ck_input)
        m_eve    = Eve(c)
        loss_ab  = ab_loss_fn(m_bits, m_bob, m_eve)
        loss_b   = bob_loss_fn(m_bits, m_bob)

    grads = tape.gradient(loss_ab, Alice.trainable_variables + Bob.trainable_variables)
    opt_AB.apply_gradients(zip(grads, Alice.trainable_variables + Bob.trainable_variables))

    # --- Train Eve (Alice frozen) ---
    m_bits_e, keys_e, _ = sample_data(BATCH_SIZE, N_BITS)
    with tf.GradientTape() as tape_e:
        mk_input_e = tf.concat([m_bits_e, keys_e], axis=1)
        c_e        = tf.stop_gradient(Alice(mk_input_e))
        m_eve      = Eve(c_e)
        loss_e     = eve_loss_fn(m_bits_e, m_eve)

    grads_e = tape_e.gradient(loss_e, Eve.trainable_variables)
    opt_E.apply_gradients(zip(grads_e, Eve.trainable_variables))

    # Logging
    loss_history_ab.append(loss_ab.numpy())
    loss_history_bob.append(loss_b.numpy())
    loss_history_eve.append(loss_e.numpy())

    if loss_ab < best_loss:
        Alice.save('training_networks/models/Alice_best.keras')
        Bob.save('training_networks/models/Bob_best.keras')
        Eve.save('training_networks/models/Eve_best.keras')
        best_loss = loss_ab
        best_loss_iteration = step + 1
        # print all the loss values
        print(f"Best step {step+1}: AB loss: {loss_ab:.4f}, Bob loss: {loss_b:.4f}, Eve loss: {loss_e:.4f} - best: {best_loss:.4f}")

    # if (step + 1) % 500 == 0:
    #     print(f"Step {step+1}: AB loss: {loss_ab:.4f}, Bob loss: {loss_b:.4f}, Eve loss: {loss_e:.4f} - best: {best_loss:.4f} at step {best_loss_iteration}")

    # I need an early stopping condition if the loss doesn't improve for PATIENCE steps then finish the loop
    if (step + 1) - best_loss_iteration > PATIENCE:
        print(f"No improvement for {PATIENCE} rounds. Stopping at step {step+1}.")
        # load the best models
        Alice.load_weights('training_networks/models/Alice_best.keras')
        Bob.load_weights('training_networks/models/Bob_best.keras')
        Eve.load_weights('training_networks/models/Eve_best.keras')
        print(f"Best loss: {best_loss:.4f} at step {best_loss_iteration}")
        break

#  Plot training history of all losses
plt.figure(figsize=(8,5))
plt.plot(loss_history_ab, label='AB loss')
plt.plot(loss_history_bob, label='Bob loss')
plt.plot(loss_history_eve, label='Eve loss')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss curves during training")
plt.grid(True)
plt.show()