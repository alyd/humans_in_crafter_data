#!/usr/bin/env python3
"""Write mean(log(counts + 1)) to a file."""

import pickle
import sys
from pathlib import Path
import numpy as np
from collections import Counter

def compute_infogain(transition_table):
    matrix = transition_table
    obs_act_matrix = {}
    for (obs, act, next_obs), count in matrix.items():
        key = (obs, act)
        if key not in obs_act_matrix:
            obs_act_matrix[key] = 0
        obs_act_matrix[key] += count

    log_sum = 0
    total = 0
    for (obs, act), count in obs_act_matrix.items():
        log_sum += np.log(count + 1)
        total += count

    if total == 0:
        return total
    log_mean = log_sum / total
    return log_mean

def compute_cu_infogain(transition_table, cu_transition_table):
    """calculate the infogain of the episode whose transitions are stored in transition_table, accounting for all the information already gained BEFORE the start of this episode from the transitions in cu_transition_table"""

    obs_act_matrix = {}
    for (obs, act, next_obs), count in transition_table.items():
        key = (obs, act)
        if key not in obs_act_matrix:
            obs_act_matrix[key] = 0
        obs_act_matrix[key] += count

    prev_cu_transition_table = Counter(cu_transition_table) - Counter(transition_table)

    cu_obs_act_matrix = {}
    for (obs, act, next_obs), count in prev_cu_transition_table.items():
        key = (obs, act)
        if key not in cu_obs_act_matrix:
            cu_obs_act_matrix[key] = 0
        cu_obs_act_matrix[key] += count

    log_sum = 0
    total = 0
    for (obs, act), count in obs_act_matrix.items():
        cu_count = 0
        if (obs, act) in cu_obs_act_matrix:
            cu_count = cu_obs_act_matrix[(obs, act)]
        log_sum += np.log(count + cu_count + 1) - np.log(cu_count + 1)
        total += count

    if total == 0:
        return total
    log_mean = log_sum / total
    return log_mean


# def compute_cu_infogain(part_tables_dict):
#     """ takes a dictionary of {run_name: transition table} from a single participant, returns a list of info gains"""
#     sorted_tables = sorted(part_tables_dict.items())
#     cu_table = Counter({})
#     info_gains = []
#     for run, table in sorted_tables:
#         cu_table = cu_table + Counter(table)
#         info_gains.append(compute_infogain(cu_table))
#     return info_gains

