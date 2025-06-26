import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from scipy import sparse
import networkx as nx
import pandas as pd
import hashlib
import glob
import tldextract
import nltk
import pickle
import numpy as np
import scipy as sp
import itertools
import random
import glob
import json
import time
import sys
import csv
import os
import re
import string
from pathlib import Path
import distutils
import shutil


def clean_nces_add_leading_zeros(leaid, schid):
    leaid = str(leaid)
    schid = str(schid)

    while len(leaid) < 7:
        leaid = "0" + leaid

    while len(schid) < 5:
        schid = "0" + schid

    return leaid + schid


def read_pkl(input_file):
    with open(input_file, "rb") as f:
        return pickle.load(f)


def write_pkl(output_file, output_obj):
    with open(output_file, "w") as f:
        pickle.dump(output_obj, f)


def read_dict(input_file):
    with open(input_file, "r") as f:
        return json.loads(f.read())


def write_dict(output_file, output_dict, indent=4):
    with open(output_file, "w") as f:
        f.write(json.dumps(output_dict, indent=indent))


def read_obj(input_file):
    with open(input_file, "r") as f:
        return eval(f.read())


def write_obj(output_file, output_arr):
    with open(output_file, "w") as f:
        f.write(str(output_arr))


"""
	Helper function to compute the jaccard index of two sets, set1 and set2.
	i.e.:
		|set1 intersect set2| / |set1 union set2|

	Inputs:
		set1 = first set
		set2 = second set

	Output:
		jaccard_index = computed using the formula above

"""


def compute_jaccard(set1, set2):
    return float(len(set1.intersection(set2))) / len(set1.union(set2))


"""
	Helper function to compute the simple overlap index of set1 and set 2.

	i.e.:
		|set1 intersect set2| / |set1|

	NOTE: this assumes that |set1| = |set2|


	Inputs:
		set1 = first set
		set2 = second set

	Output:
		overlap_index = computed using the formula above

"""


def compute_overlap(set1, set2):
    return float(len(set1.intersection(set2))) / len(set1)


"""
	Normalizes an array
"""


def normalize_array(arr):
    return np.divide(arr, float(np.sum(arr)))


"""
	Compute the symmetric kl divergence between two distributions.  Skips values that are 0.
	Inputs:
		p_k, q_k = distributions we'd like to compute the kl-divergence between

	Outputs:
		kl_divergence = I wonder what this is ...
"""


def symmetric_kl_divergence(p_k, q_k):
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


"""
	Compute the kl divergence between two distributions.  Skips values that are 0.
	Inputs:
		p_k, q_k = distributions we'd like to compute the kl-divergence between

	Outputs:
		kl_divergence = I wonder what this is ...
"""


def kl_divergence(p_k, q_k):
    p_k = normalize_array(p_k)
    q_k = normalize_array(q_k)

    num_items = len(p_k)
    total = 0
    for i in range(0, num_items):
        if q_k[i] == 0 or p_k[i] == 0:
            continue
        total += p_k[i] * (np.log(p_k[i]) - np.log(q_k[i]))
    return total


"""
	Hellinger distance ...
"""


def hellinger(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)


"""
	Parses a URL and returns the domain
"""


def get_domain_from_url(url):
    parse_result = urlparse.urlparse(url)
    tld_result = tldextract.extract(url)
    return tld_result.domain


"""
	Returns a hash for an input string
"""


def get_hash(input_str):
    hash_obj = hashlib.sha224(input_str)
    hex_digest = hash_obj.hexdigest()
    return hex_digest


"""
	Removes non ascii characters from string
"""


def remove_non_ascii(text):
    return "".join([i if ord(i) < 128 else " " for i in text])


"""
	Checks if a word is ascii or not
"""


def is_ascii(word):
    check = string.ascii_letters + "."
    if word not in check:
        return False
    return True


"""
	From: https://stackoverflow.com/questions/39512260/calculating-gini-coefficient-in-python-numpy
"""


def gini(x):
    """Compute Gini coefficient of array of values"""
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
