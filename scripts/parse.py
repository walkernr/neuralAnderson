# -*- coding: utf-8 -*-
"""
Created on Mon Jul 02 21:24:12 2018

@author: Nicholas
"""

import argparse
import os
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
if '400spin' in CWD:
    NF = 400
if '512spin' in CWD:
    NF = 512
if '600spin' in CWD:
    NF = 600
# data file listing
FLS = [file for file in os.listdir(DIR) if '.dat' in file]
# r-values (parameter for density of states)
R = np.array([np.float32(FLS[i][:4]) for i in range(len(FLS))])
# dump r-values and feature time domain
np.save(CWD+'/sga.dat.npy', R)
np.save(CWD+'/sga.dmp.t.npy', np.arange(NF, dtype=np.int16))
if VERBOSE:
    print('r parameters and feature time domain dumped')
# parse data
if VERBOSE:
    DMP = np.array([np.loadtxt(DIR+FLS[i], dtype=np.int16)[NF:, 1].reshape(-1, NF) \
                    for i in tqdm(range(len(FLS)))])
else:
    DMP = np.array([np.loadtxt(DIR+FLS[i], dtype=np.int16)[NF:, 1].reshape(-1, NF) \
                    for i in range(len(FLS))])
# dump data
np.save(CWD+'/sga.dmp.npy', DMP)
if VERBOSE:
    print('feature data dumped')
