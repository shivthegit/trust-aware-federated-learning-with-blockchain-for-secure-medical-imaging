import torch

def fedavg(state_dicts):
    avg_state = {}

    for key in state_dicts[0].keys():
        avg_state[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)

    return avg_state
