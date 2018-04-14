from math import floor


age_bin_probs = [
    0.065, 0.066, 0.067, 0.071, 0.070, 0.068, 0.065, 0.065, 0.068,
    0.074, 0.072, 0.064, 0.054, 0.040, 0.030, 0.024, 0.019, 0.018]


def age_idx(x, base=5):
    return int(floor(x/base))


def age_mult(age):
    return age_bin_probs[age_idx(age)]


race_mult = {
    'American Indian': 0.00,
    'Asian': 0.20,
    'Black': -0.14,
    'Pacific Islander': -0.03,
    'White': 0.23,
    'mixed_other': -0.03
    }

skin_mult = {
    'normal': 0.00,
    'oily': 0.3,
    'dry': -0.3
    }


def get_multiplier(person):
    ret = 1
    ret += age_mult(person['age'])
    ret += race_mult[person['race']]
    ret += skin_mult[person['skin']]
    return ret
