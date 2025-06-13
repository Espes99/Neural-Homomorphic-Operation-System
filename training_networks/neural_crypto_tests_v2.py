
import logging
import os
import tensorflow as tf
import numpy as np
import matplotlib.ticker as mtick

# -----------------------------
# Configuration
# -----------------------------
N_BITS      = 32
N_ROUNDS    = 50000000   # HO training iterations
BATCH_SIZE  = 1024
MODEL_DIR   = 'training_networks/models'
ALICE_MODEL = os.path.join(MODEL_DIR, 'Alice_best_100.keras')
BOB_MODEL   = os.path.join(MODEL_DIR, 'Bob_best_100.keras')
EVE_MODEL = os.path.join(MODEL_DIR, 'Eve_best_100.keras')
FONT_SIZE = 16
# Suppress TensorFlow info and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# createa float to bit conversion like a binary representation but use integer to binary conversion before that multiply the floating number with a big number
# to get the integer representation
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

ho_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'HO_best_100.keras'))
# -----------------------------
# Load frozen Alice & Bob
# -----------------------------
Alice = tf.keras.models.load_model(ALICE_MODEL, safe_mode=False)
Bob   = tf.keras.models.load_model(BOB_MODEL)
Eve = tf.keras.models.load_model(EVE_MODEL)
Alice.trainable = False
Bob.trainable   = False
Eve.trainable   = False
print("Loaded HO model from disk.")

NUMBER_OF_SAMPLES = 20
bits1, bits2, c1, c2, key, bits_sum = sample_data_for_ho(NUMBER_OF_SAMPLES, N_BITS)

ho_in = tf.concat([c1, c2, key], axis=1)
c3 = ho_model(ho_in)
m3 = Bob(tf.concat([c3, key], axis=1)).numpy()
m3_int = np.round(m3).astype(np.float32)
m3_eve = Eve(c3)
print("\nHO example (20 samples):")
# print them in a pandas table with the following columns:
# Float 1, Float 2, sum, Predicted Sum, MAE
import pandas as pd
df = pd.DataFrame()
df['Float 1'] = bits_to_float(bits1)
df['Float 2'] = bits_to_float(bits2)
df['sum'] = bits_to_float(bits_sum)
df['Predicted Sum'] = bits_to_float(m3_int)
df['MAE'] = np.abs(df['sum'] - df['Predicted Sum'])
# order by the MAE
df = df.sort_values(by='MAE', ascending=True)
print(df.to_string(index=False))


# perform this experiments for 1000 samples and plot the histogram of the errors
import matplotlib.pyplot as plt
import seaborn as sns



bits1, bits2, c1, c2, key, bits_sum = sample_data_for_ho(5000, N_BITS) 
ho_in = tf.concat([c1, c2, key], axis=1)
c3 = ho_model(ho_in)
m3 = Bob(tf.concat([c3, key], axis=1)).numpy()
m3_int = np.round(m3).astype(np.float32)
eve_decrypted = Eve(c3)
errors = np.abs(bits_to_float(bits_sum) - bits_to_float(m3_int))
errors_eve = np.abs(bits_to_float(bits_sum) - bits_to_float(eve_decrypted.numpy()))
#print error for eve
print(f"Mean Absolute Error (Bob): {np.mean(errors):.8f}")
print(f"Mean Absolute Error (Eve): {np.mean(errors_eve):.8f}")

# Generate histogram for Eve's decryption errors
p50_eve = np.percentile(errors_eve, 50)
p75_eve = np.percentile(errors_eve, 75)

plt.figure(figsize=(10, 6))
sns.histplot(errors_eve, bins=50, kde=True, stat='probability', alpha=0.6, color='g', edgecolor='black')

# Add vertical lines
plt.axvline(p50_eve, color='blue', linestyle='--', linewidth=2, label='50th Percentile')
plt.axvline(p75_eve, color='red', linestyle='--', linewidth=2, label='75th Percentile')

plt.xlabel('Error', fontsize=FONT_SIZE)
plt.ylabel('Frequency', fontsize=FONT_SIZE)
plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE-2)
plt.legend(fontsize=FONT_SIZE)
plt.grid(True)

# Save the figure
plt.savefig('training_networks/eve_error_histogram.png', dpi=300, bbox_inches='tight')
plt.show()

p50 = np.percentile(errors, 50)
p75 = np.percentile(errors, 75)

plt.figure(figsize=(10, 6))
# y axis should b the percentage of samples using sns.histplot
# plt.hist(errors, bins=50, density=True, alpha=0.6, color='g', edgecolor='black')
# plt.hist(errors, bins=50, density=True, alpha=0.6, color='g', edgecolor='black')
sns.histplot(errors, bins=50, kde=True, stat='probability',alpha=0.6, color='g', edgecolor='black')

# Add vertical lines
plt.axvline(p50, color='blue', linestyle='--', linewidth=2, label='50th Percentile')
plt.axvline(p75, color='red', linestyle='--', linewidth=2, label='75th Percentile')

plt.xlabel('Error', fontsize=FONT_SIZE)
plt.ylabel('Frequency', fontsize=FONT_SIZE)
plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE-2)
plt.legend(fontsize=FONT_SIZE)
plt.grid(True)
plt.savefig('training_networks/ho_error_histogram.png', dpi=300, bbox_inches='tight')
plt.show()