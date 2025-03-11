import hashlib
import random
import string


def generate_hash_from_string(text: str):
    """Generate a hash from a string."""
    hash_object = hashlib.sha256(text.encode())
    hash_string = hash_object.hexdigest()
    return hash_string


def generate_random_hash():
    """Generate a random hash."""
    random_string = "".join(random.choices(string.ascii_letters + string.digits, k=32))

    return generate_hash_from_string(random_string)
