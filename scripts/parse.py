# -*- coding: utf-8 -*-
"""
Created on Mon Jul 02 21:24:12 2018

@author: Nicholas
"""

import argparse
import sys
import os
import pickle
import numpy as np
from tqdm import tqdm

def parse_args():
    ''' parse command line arguments '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='verbose output', action='store_true')
    args = parser.parse_args()
    return args.verbose

VERBOSE = parse_args()
# current working directory
CWD = os.getcwd()

# system data directory and feature count
DIR = CWD+'/data/'
if 'b100' in CWD:
    NF = 400
if 'b150' in CWD:
    NF = 600
# data file listing
FLS = os.listdir(DIR)
# r-values (parameter for density of states)
R = np.array([np.float32(FLS[i][6:10]) for i in range(len(FLS))])
# dump r-values and feature time domain
pickle.dump(R, open(CWD+'/sga.r.pickle', 'wb'))
pickle.dump(np.arange(NF, dtype=np.int16), open(CWD+'/sga.spin.t.pickle', 'wb'))
if VERBOSE:
    print('r parameters and feature time domain dumped')
# parse data
if VERBOSE:
    DAT = np.array([np.loadtxt(DIR+FLS[i], dtype=np.int16)[NF:, 1].reshape(-1, NF) for i in tqdm(range(len(FLS)))])
else:
    DAT = np.array([np.loadtxt(DIR+FLS[i], dtype=np.int16)[NF:, 1].reshape(-1, NF) for i in range(len(FLS))])
# dump data
pickle.dump(DAT, open(CWD+'/sga.spin.pickle', 'wb'))
if VERBOSE:
    print('feature data dumped')