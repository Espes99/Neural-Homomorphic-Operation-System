import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from matplotlib import pyplot as plt

# Suppress TensorFlow info and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# -----------------------------
# Configuration
# -----------------------------
N_BITS      = 32
N_ROUNDS    = 1000000000   # HO training iterations
BATCH_SIZE  = 4096*16
MODEL_DIR   = 'training_networks/models'
ALICE_MODEL = os.path.join(MODEL_DIR, 'Alice_best_100.keras')
BOB_MODEL   = os.path.join(MODEL_DIR, 'Bob_best_100.keras')

# -----------------------------
# Load frozen Alice & Bob
# -----------------------------
Alice = tf.keras.models.load_model(ALICE_MODEL, safe_mode=False)
Bob   = tf.keras.models.load_model(BOB_MODEL)
Alice.trainable = False
Bob.trainable   = False

# -----------------------------
# Helper: float <-> bits
# -----------------------------

# createa float to bit conversion like a binary representation but use integer to binary conversion before that multiply the floating number with a big number
# to get the integer representation
# scale is 500000
SCALE = 100_000

def float_to_bits(x):
    """
    Convert scalar or 1D array of floats into bit‐vectors of
    their scaled‐to‐int representation.
    """
    x_arr = np.atleast_1d(np.array(x, dtype=np.float32))
    # Round to nearest scaled integer
    x_int = np.round(x_arr * SCALE).astype(np.int32)             # shape (n,)
    # Extract bits: (n,32), big‐endian bit order
    bits = ((x_int[:, None] >> np.arange(N_BITS-1, -1, -1)) & 1).astype(np.uint8)
    return bits.astype(np.float32)  # if you need floats

def bits_to_float(bits):
    """
    Convert bit‐vectors back into floats by reconstructing the
    scaled integer and then dividing by SCALE.
    """
    # bits: (n,32) array of 0/1
    # Compute integer = sum bits * 2**(31-i)
    weights = (2 ** np.arange(N_BITS-1, -1, -1)).astype(np.int32)      # shape (32,)
    ints    = bits.astype(np.int32) @ weights                   # shape (n,)
    return ints.astype(np.float32) / SCALE

def bits_to_float_tf(bits):
    """
    TensorFlow version of bits_to_float for tensor inputs.
    Input: bits (batch_size, 32) tensor of 0/1 values.
    Output: float tensor (batch_size,) of reconstructed floats.
    """
    # Weights for bits: 2^(31), 2^(30), ..., 2^0
    weights = tf.constant(2 ** np.arange(N_BITS-1, -1, -1), dtype=tf.float32)  # shape (32,)
    # Convert bits to float32
    bits = tf.cast(bits, tf.float32)  # Ensure bits are float32
    # Compute weighted sum: bits * weights, then sum along axis 1
    ints = tf.reduce_sum(bits * weights, axis=1)  # shape (batch_size,)
    return ints / SCALE

def test_float_to_bits():
    for i in range(10):
        x = np.random.rand(1).astype(np.float32)
        bits = float_to_bits(x)
        recovered_x = bits_to_float(bits)
        print(f"Original: {x}, Recovered: {recovered_x}, diff: {np.abs(x - recovered_x)}")
    print("Test completed.")
test_float_to_bits()

# -----------------------------
# Generate HO training data
# -----------------------------
def sample_data_for_ho(batch_size, n_bits):
    # two random floats
    x1 = np.random.rand(batch_size).astype(np.float32)
    x2 = np.random.rand(batch_size).astype(np.float32)

    x3 = x1 + x2

    # convert to bit-vectors
    bits1 = float_to_bits(x1)
    bits2 = float_to_bits(x2)

    bits_sum = float_to_bits(x3)

    # single shared key
    key = np.random.randint(0, 2, (batch_size, n_bits)).astype(np.float32)
    # produce ciphertexts via Alice
    inp1 = np.concatenate([bits1, key], axis=1)
    inp2 = np.concatenate([bits2, key], axis=1)
    c1 = Alice(inp1)
    c2 = Alice(inp2)

    return bits1, bits2, c1.numpy(), c2.numpy(), key, bits_sum

# -----------------------------
# Build HO model
# -----------------------------
def make_ho_model(mult_factor=1.0):
    inp = tf.keras.Input(shape=(N_BITS*3,))  # c1 (32) + c2 (32) + key (32)
    x = tf.keras.layers.Dense(int(256*mult_factor), activation='relu')(inp) # tf.keras.layers.Dense(256, activation='sigmoid')(inp)
    # add a dropout layer
    x = tf.keras.layers.Dropout(0.2)(x)
    # x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(int(256*mult_factor), activation='relu')(x) # tf.keras.layers.Dense(256, activation='sigmoid')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(N_BITS, activation='sigmoid')(x)  # new ciphertext
    return tf.keras.Model(inputs=inp, outputs=out)

ho_model = make_ho_model(mult_factor=2.0)

# if model exists, load it
if os.path.exists(os.path.join(MODEL_DIR, 'HO_best.keras')):
    ho_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'HO_best.keras'))
    print("Loaded HO model from disk.")

opt = tf.keras.optimizers.legacy.Adam(1e-3)

# 1) define your schedule
lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-3,   # start at 0.001
    decay_steps=100_000,          # every 100k steps
    decay_rate=0.5,               # multiply LR by 0.5
    staircase=True,               # do discrete drops
)

# 2) pass it into your optimizer
# opt = Adam(learning_rate=lr_schedule, amsgrad=True)
# opt = AdamW(
#     learning_rate=lr_schedule,   # your existing schedule  
#     weight_decay=1e-5,           # e.g. 1×10⁻⁵
#     amsgrad=True
# )


# mean absolute error
loss_func_ho = tf.keras.losses.MeanAbsoluteError()

# -----------------------------
# Training Loop for HO
# -----------------------------
loss_history = []
best_loss = np.inf
best_step = 0

PATIENCE = 5000000

for step in tqdm(range(N_ROUNDS), desc='Training AB+HO'):
    bits1, bits2, c1, c2, key, bits_sum = sample_data_for_ho(BATCH_SIZE, N_BITS)

    # save the first step data for testing
    if step == 0:
        bits1_t, bits2_t, c1_t, c2_t, key_t, sum_t = bits1, bits2, c1, c2, key, bits_sum

    # Prepare tensors
    bits1_tf = tf.convert_to_tensor(bits1)
    bits2_tf = tf.convert_to_tensor(bits2)
    c1_tf    = tf.convert_to_tensor(c1)
    c2_tf    = tf.convert_to_tensor(c2)
    key_tf   = tf.convert_to_tensor(key)
    bits_sum_tf = tf.convert_to_tensor(bits_sum)

    with tf.GradientTape() as tape:
        # Concatenate c1, c2, key to input for HO
        ho_in = tf.concat([c1_tf, c2_tf, key_tf], axis=1)
        c3 = ho_model(ho_in)
        # Decrypt with Bob
        m3_logits = Bob(tf.concat([c3, key_tf], axis=1))
        # Compute MSE loss against true bit-sum
        # convert bits to float
        bits_sum_tf = bits_to_float_tf(bits_sum_tf)
        # convert m3_logits to float
        m3_logits = bits_to_float_tf(m3_logits)
        # compute loss
        # use mean absolute error
        loss = loss_func_ho(bits_sum_tf, m3_logits)
        #loss = loss_func_ho(bits_sum_tf, m3_logits)

    grads = tape.gradient(loss, ho_model.trainable_variables)
    opt.apply_gradients(zip(grads, ho_model.trainable_variables))
    loss_history.append(loss.numpy())

    # Save best
    if loss < best_loss:
        best_loss = loss
        ho_model.save(os.path.join(MODEL_DIR, 'HO_best.keras'))
        best_step = step
        print(f"Best step {step+1}: best_loss = {best_loss:.6f}")

    # if no improvement for PATIENCE steps, stop training
    if (step+1) - best_step > PATIENCE:
        print(f"No improvement for {PATIENCE} rounds. Stopping at step {step+1} with the best step {best_step} and loss {best_loss:.4f}.")
        # load the best model
        ho_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'HO_best.keras'))
        break
#  Plot training history of all losses
plt.figure(figsize=(8,5))
plt.plot(loss_history, label='HO loss')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss curves during training - HO")
plt.grid(True)
plt.savefig("ho_loss_curve.png")
plt.show()

ho_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'HO_best.keras'))
# -----------------------------
# Example: test HO model
# -----------------------------
bits1, bits2, c1, c2, key, bits_sum = sample_data_for_ho(5, N_BITS)

ho_in = tf.concat([c1, c2, key], axis=1)
c3 = ho_model(ho_in)
m3 = Bob(tf.concat([c3, key], axis=1)).numpy()
m3_int = np.round(m3).astype(np.float32)

print("\nHO example (5 samples):")
for i in range(5):
    print("-"*30)
    print("bits1     :", bits1[i][:8])
    print("bits2     :", bits2[i][:8])
    print(f"bit-sum  : {bits_sum[i][:8]}")
    print(f"predicted: {m3_int[i][:8]}")
    print(f"pred: {m3[i][:8]}, int: {m3_int[i][:8]}")
    print(f"float1: {bits_to_float(bits1[i])}, float2: {bits_to_float(bits2[i])}")
    print(f"sum: {bits_to_float(bits_sum[i])}")
    print(f"float of m3_int: {bits_to_float(m3_int[i])}")

os.makedirs(MODEL_DIR, exist_ok=True)
