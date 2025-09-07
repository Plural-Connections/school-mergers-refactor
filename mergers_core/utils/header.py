import hashlib
import tldextract
import pickle
import numpy as np
import json
import os
import string


def read_pkl(input_file: os.PathLike):
    with open(input_file, "rb") as f:
        return pickle.load(f)


def write_pkl(output_file: os.PathLike, output_obj) -> None:
    with open(output_file, "w") as f:
        pickle.dump(output_obj, f)








def read_obj(input_file: os.PathLike):
    with open(input_file, "r") as f:
        return eval(f.read())


def write_obj(output_file: os.PathLike, output_obj):
    with open(output_file, "w") as f:
        f.write(str(output_obj))


def compute_jaccard(set1, set2):
    """Helper function to compute the jaccard index of two sets, set1 and set2.
    i.e.:
        |set1 intersect set2| / |set1 union set2|

    Inputs:
        set1 = first set
        set2 = second set

    Output:
        jaccard_index = computed using the formula above
    """
    return float(len(set1.intersection(set2))) / len(set1.union(set2))


def compute_overlap(set1, set2):
    """Helper function to compute the simple overlap index of set1 and set 2.
    i.e.:
        |set1 ∩ set2| / |set1|
    NOTE: this assumes that |set1| = |set2|

    Inputs:
        set1 = first set
        set2 = second set

    Output:
        overlap_index = computed using the formula above
    """
    return len(set1.intersection(set2)) / len(set1)


def normalize_array(arr):
    """Normalizes an array."""
    return np.divide(arr, float(np.sum(arr)))


def symmetric_kl_divergence(p_k, q_k):
    """Compute the symmetric kl divergence between two distributions.  Skips values that are 0.
    Inputs:
        p_k, q_k = distributions we'd like to compute the kl-divergence between

    Outputs:
        kl_divergence = I wonder what this is ...
    """
    p_k = normalize_array(p_k)
    q_k = normalize_array(q_k)

    num_items = len(p_k)
    total = 0
    for i in range(0, num_items):
        if q_k[i] == 0 or p_k[i] == 0:
            continue
        curr = p_k[i] * (np.log(p_k[i]) - np.log(q_k[i])) + q_k[i] * (
            np.log(q_k[i]) - np.log(p_k[i])
        )
        total += curr
    return total


def kl_divergence(p_k, q_k):
    """Compute the kl divergence between two distributions.  Skips values that are 0.
    Parameters:
        p_k, q_k = distributions we'd like to compute the kl-divergence between
    """
    p_k = normalize_array(p_k)
    q_k = normalize_array(q_k)

    num_items = len(p_k)
    total = 0
    for i in range(0, num_items):
        if q_k[i] == 0 or p_k[i] == 0:
            continue
        total += p_k[i] * (np.log(p_k[i]) - np.log(q_k[i]))
    return total


def hellinger(p, q):
    """Hellinger distance."""
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)


def get_domain_from_url(url):
    """Parses a URL and returns the domain."""
    tld_result = tldextract.extract(url)
    return tld_result.domain


def get_hash(input_str):
    """Returns a hash for an input string."""
    hash_obj = hashlib.sha224(input_str)
    hex_digest = hash_obj.hexdigest()
    return hex_digest


def remove_non_ascii(text):
    """Removes non ascii characters from string."""
    return "".join([i if ord(i) < 128 else " " for i in text])


def is_ascii(word):
    """Checks if a word is ascii or not."""
    check = string.ascii_letters + "."
    if word not in check:
        return False
    return True


def gini(x):
    """Compute Gini coefficient of array of values.

    Credit: https://stackoverflow.com/questions/39512260/calculating-gini-coefficient-in-python-numpy
    """
    x = np.array(x)
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x) ** 2 * np.mean(x))


def add_leading_zero_for_school(series):
    return series.str.rjust(12, "0")


def add_leading_zero_for_district(series):
    return series.str.rjust(7, "0")


def update_dist_id_with_leading_zero(dist_id):
    return dist_id.rjust(7, "0")
