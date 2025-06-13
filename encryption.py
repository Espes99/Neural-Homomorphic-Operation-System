import tenseal as ts
import numpy as np

def create_ckks_context() -> ts.context:
    """
    Create and return a CKKS context (TenSEAL) with default parameters.
    """
    ctx = ts.context(
        scheme=ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        # original: [60, 40, 40, 60]
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    # Global scale
    ctx.global_scale = 2**40
    # Generate Galois keys for efficient vector operations
    ctx.generate_galois_keys()
    return ctx

def encrypt_vector(context: ts.context, vector: np.ndarray) -> ts.CKKSVector:
    """
    Encrypt a 1D NumPy array with CKKS.
    """
    encrypted = ts.ckks_vector(context, vector.tolist())
    return encrypted


def decrypt_vector(context: ts.context, encrypted_vector: ts.CKKSVector) -> np.ndarray:
    """
    Decrypt a CKKSVector to a NumPy array.    """
    decrypted = np.array(encrypted_vector.decrypt())
    return decrypted


def load_CKKSVector_from_buffer(context: ts.context, buffer: bytes) -> ts.CKKSVector:
    """
    Load a CKKSVector from a buffer.
    """
    vector_from_buffer = ts.ckks_vector_from(context, buffer)

    return vector_from_buffer
