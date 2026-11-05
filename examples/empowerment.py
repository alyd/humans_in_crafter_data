#!/usr/bin/env python3
"""Calculate n-step empowerment over the lifetime of an agent by multiplying together copies of the combined transition matrix. Follows the implementation from https://github.com/Mchristos/empowerment"""

import pickle
import sys
import numpy as np
import sparse
import itertools
from functools import reduce

import numpy as np 
from numpy import log2

import preprocessing

eps = 1e-40

def _rand_dist(shape):
    """ define a random probability distribution """
    P = np.random.rand(*shape)
    return _normalize(P)

def _normalize(P):
    """ normalize probability distribution """
    s = sum(P)
    if s == 0.:
        raise ValueError("input distribution has sum zero")
    return P / s


def blahut_arimoto(P_yx, epsilon = 0.001, deterministic = False):
    """ 
    Compute the channel capacity C of a channel p(y|x) using the Blahut-Arimoto algorithm. To do
    this, finds the input distribution p(x) that maximises the mutual information I(X;Y)
    determined by p(y|x) and p(x).

    P_yx : defines the channel p(y|x)
    iters : number of iterations
    """
    P_yx = P_yx + eps
    if not deterministic:
        # initialize input dist randomly 
        q_x = _rand_dist((P_yx.shape[1],))
        T = 1
        while T > epsilon:
            # update PHI
            PHI_yx = (P_yx*q_x.reshape(1,-1))/(P_yx @ q_x).reshape(-1,1)
            r_x = np.exp(np.sum(P_yx*log2(PHI_yx), axis=0))
            # channel capactiy 
            C = log2(np.sum(r_x))
            # check convergence 
            T = np.max(log2(r_x/q_x)) - C
            # update q
            q_x = _normalize(r_x + eps)
        if C < 0:
            C = 0
        return C
    else:
        # assume all columns in channel matrix are peaked on a single state
        # log of number of reachable states
        return log2(np.sum(P_yx.sum(axis=1) > 0.999))

def dense_indices(encodings):
    """Map the very large numbers returned by encode to unique indices."""
    encoding_to_index = {encoding: index for index, encoding in enumerate(set(encodings))}
    return encoding_to_index

def compute_empowerment(transition_table, n_step: int=1, n_samples:int=50, approximate='blahut', epsilon=1e-6,):
    
    # transition table is T(s,a,s') as counts, first we need to normalize into probabilities
    # Get the keys and values from the dictionary and construct a sparse array
    if len(transition_table) == 0:
        return 0
    coords, values = [ls for ls in zip(*transition_table.items())] 
    
    all_obs = set()
    for ob1, ob2 in zip([c[0] for c in coords], [c[2] for c in coords]):
        all_obs.add(ob1)
        all_obs.add(ob2)

    encoding_to_index = dense_indices(all_obs)

    encoded_coords = np.zeros((len(coords),3), dtype=np.int32)
    for i, (ob1, ob2) in enumerate(zip([c[0] for c in coords], [c[2] for c in coords])):
        encoded_coords[i,0] = int(encoding_to_index[ob1])
        encoded_coords[i,1] = int(coords[i][1])
        encoded_coords[i,2] = int(encoding_to_index[ob2])
        # import pdb; pdb.set_trace()
        # encoded_coords[i] = tuple(encoded_coords[i])

    obs_dim = encoded_coords[:, (0, 2)].max() + 1
    act_dim = encoded_coords[:, 1].max() + 1
    transition_tensor = sparse.COO(encoded_coords.T, np.array(values).astype(np.float32), shape=(obs_dim, act_dim, obs_dim)) # construct T(s,a,s') as (S, A, S')
    transition_tensor_probs = transition_tensor / transition_tensor.sum(axis=2, keepdims=True) # P(s'|s,a), where s' is the last axis

    n_states, n_actions, _  = transition_tensor_probs.shape 
        # sample action sequences
    if approximate == 'count':
        if n_actions**n_step < 5000:
            nstep_action_samples = np.array(list(itertools.product(range(n_actions), repeat = n_step)))
        else:
            nstep_action_samples = np.random.randint(0,n_actions, [n_samples, n_step] )
        
        total_empowerment = 0
        T = transition_tensor_probs.todense()
        T = np.nan_to_num(T)
            # counting future states
            # fold over each nstep actions, get unique end states
        tmap = lambda s, a : np.argmax(T[s, a, :]) 
        seen = set()
        for state in range(n_states):
            for i in range(len(nstep_action_samples)):
                aseq = nstep_action_samples[i,:]
                seen.add(reduce(tmap, [state, *aseq]))
        # empowerment = log # of unique reachable states 
            total_empowerment += np.log2(len(seen))
        return total_empowerment      
    elif approximate == 'blahut':  
        T = transition_tensor_probs.todense()
        T = np.nan_to_num(T)
        nstep_action_samples = list(itertools.product(range(n_actions), repeat = n_step))
        Bn = np.zeros([n_states, len(nstep_action_samples), n_states])
        for i, an in enumerate(nstep_action_samples):
            Bn[:, i , :] = reduce((lambda x, y : np.dot(y, x)), map((lambda a : T[:,a,:]), an))
        total_empowerment = 0
        for state in range (n_states):
            total_empowerment += blahut_arimoto(Bn[state,:,:], epsilon=epsilon)
        return total_empowerment

    # single step
    else:
        pass

## FOR DEBUGGGGGING
# dataset_paths=['../expl_in_crafter/datasets/kids_scored_dataset']

# from dataloading_utils import load_into_df
# dataset_df = load_into_df(dataset_paths)
# dataset_df = preprocessing.add_transition_tables(dataset_df)
# empowerment_vals=[]
# for ep in dataset_df.itertuples():
#     print(ep.participant, ep.run)
#     empowerment_vals.append(compute_empowerment(ep.transition_table))
#     import pdb; pdb.set_trace()
# dataset_df['empowerment_vals'] = empowerment_vals