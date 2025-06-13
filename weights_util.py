import numpy as np
import os
import tensorflow as tf
from encryption import encrypt_vector, load_CKKSVector_from_buffer, decrypt_vector
from learning_params import SCALE
def encrypt_model_weights(weights, ckks_context):
    encrypted_weights = []
    original_shape = []
    for w in weights:
        original_shape.append(w.shape)
        flattened = w.flatten()
        encrypted_weight = encrypt_vector(ckks_context, flattened)
        encrypted_weights.append(encrypted_weight)
    return encrypted_weights, original_shape


def decrypt_model_weights(encrypted_weights, original_shapes, ckks_context):
    decrypted_weights = []

    for w, shape in zip(encrypted_weights, original_shapes):
        w_ser = w.serialize()
        encrypted_weight = load_CKKSVector_from_buffer(ckks_context, w_ser)
        decrypted_weight = decrypt_vector(ckks_context, encrypted_weight)
        decrypted_weight = decrypted_weight.reshape(shape)
        decrypted_weights.append(decrypted_weight)

    return decrypted_weights

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

# Load pretrained Alice and Bob weights once at import
MODEL_DIR   = 'models'
ALICE_MODEL = os.path.join(MODEL_DIR, 'Alice_best_100.keras')
BOB_MODEL   = os.path.join(MODEL_DIR, 'Bob_best_100.keras')
# Paths to your pretrained models

Alice = tf.keras.models.load_model(ALICE_MODEL)
Bob   = tf.keras.models.load_model(BOB_MODEL)

def encrypt_model_weights_ho(weights, keys):
    encrypted_weights = []
    original_shapes = []
    for i, w in enumerate(weights):
        original_shapes.append(w.shape)
        flat = w.flatten()
        key = keys[i]
        # Convert floats to bit representations
        bits = float_to_bits(flat)  # shape (n, 32)
        # Sample random key bits for each element
        # Concatenate message bits and keys, then encrypt via Alice
        mk_input = np.concatenate([bits, key], axis=1)
        ciphers = Alice(mk_input).numpy()  # shape (n, 32)
        encrypted_weights.append(ciphers)
    return encrypted_weights, original_shapes


def decrypt_model_weights_ho(encrypted_weights, original_shapes, keys):
    decrypted_weights = []
    for i, ciphers in enumerate(encrypted_weights):
        key = keys[i]
        # Concatenate ciphertext and keys, then decrypt via Bob
        ck_input = np.concatenate([ciphers, key], axis=1)
        m_bob = Bob(ck_input).numpy()  # shape (n, 32)
        # Threshold to bits
        m_bob_bin = (m_bob > 0.5).astype(np.float32)
        # Convert bits back to floats
        flat = bits_to_float(m_bob_bin)

        decrypted = flat.reshape(original_shapes[i])
        decrypted_weights.append(decrypted)
    return decrypted_weights


def add_encrypted_weights_ho(enc1, enc2, ho_model, keys):
    """
    Adds two lists of encrypted weights layer-wise using HO network.
    enc1, enc2: lists of (ciphertexts, keys) tuples.
    Returns a new list of summed encrypted weights.
    """
    results = []
    for i, (c1, c2) in enumerate(zip(enc1, enc2)):
        # Get batch-sized global key
        key = keys[i]

        # Concatenate ciphertexts and key for HO input
        ho_input = np.concatenate([c1, c2, key], axis=1)

        # Apply homomorphic operation
        result_c = ho_model(ho_input).numpy()
        results.append(result_c)

    return results

    # return [add_two_encrypted(e1, e2, ho_model) for e1, e2 in zip(enc1, enc2)]


def apply_masking_ho(old_client_weights, new_global_weights, round_num, total_rounds,
                  threshold_start=0.99, threshold_end=0.01):
    """
    Apply threshold-based masking to filter out noise from HO operations.

    Args:
        old_client_weights: List of weight lists, one per client (before aggregation)
        new_global_weights: New global weights after homomorphic aggregation
        round_num: Current round number (0-indexed)
        total_rounds: Total number of rounds
        threshold_start: Initial threshold (permissive)
        threshold_end: Final threshold (restrictive)

    Returns:
        masked_weights: List of masked weight lists, one per client
    """
    # Calculate adaptive threshold
    thr = threshold_start - (threshold_start - threshold_end) * (round_num / total_rounds)

    print(f'\n[MASKING] Round {round_num + 1}: threshold = {thr:.4f}')

    masked_weights = []
    rejection_stats = []
    overall_total = 0
    overall_kept_old = 0

    # Process each client
    for client_idx, client_old_weights in enumerate(old_client_weights):
        filtered_weights = []
        client_kept_old = 0
        client_total = 0
        client_rejected = 0
        print(f'\n-- Client {client_idx + 1} Masking --')

        # Process each layer
        for layer_idx, (w_old, w_new) in enumerate(zip(client_old_weights, new_global_weights)):
            # Calculate difference between old local and new global weights
            diff = np.abs(w_new - w_old)

            # Create mask: True where difference is within threshold (accept update)
            mask = diff <= thr

            # Count statistics
            num_accepted = np.sum(mask)  # Updates we accept
            num_rejected = np.sum(~mask)  # Updates we reject (keep old)
            client_rejected += num_rejected
            client_kept_old += num_rejected
            client_total += w_old.size
            print(f"  Layer {layer_idx + 1}: accepting {num_accepted}/{w_old.size} "
                  f"updates ({100 * num_accepted / w_old.size:.1f}%), "
                  f"rejecting {num_rejected}")

            # Apply mask: use new weight where mask is True, old weight where False
            filtered_layer = np.where(mask, w_new, w_old)
            filtered_weights.append(filtered_layer)

            # Optional: Show examples of rejected updates (likely noise)
            if num_rejected > 0 and num_rejected < 10:
                rejected_indices = np.argwhere(~mask)[:5]  # Show up to 5 examples
                print(f"    Sample rejected updates:")
                for idx in rejected_indices:
                    idx_tuple = tuple(idx)
                    print(f"      index {idx_tuple}: {w_old[idx_tuple]:.4f} → {w_new[idx_tuple]:.4f} "
                          f"(diff: {diff[idx_tuple]:.4f})")

        masked_weights.append(filtered_weights)
        overall_kept_old += client_kept_old
        overall_total += client_total
        percentage_rejected = (client_rejected / client_total) * 100 if client_total > 0 else 0.0
        print(f"  Client {client_idx + 1} summary: rejected {client_kept_old}/{client_total} "
              f"updates ({100 * client_kept_old / client_total:.1f}% kept old)")

        rejection_stats.append({
            'round': round_num + 1,
            'client': client_idx + 1,
            'total_weights': client_total,
            'rejected_weights': client_rejected,
            'rejection_percentage': percentage_rejected,
            'threshold': thr
        })

    print(f"\n[MASKING] Overall summary: rejected {overall_kept_old}/{overall_total} "
          f"updates ({100 * overall_kept_old / overall_total:.1f}%\n")

    return masked_weights, rejection_stats