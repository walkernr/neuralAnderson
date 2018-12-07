# -*- coding: utf-8 -*-
"""
Created on Mon Jul 02 21:24:48 2018

@author: Nicholas
"""

import argparse
import os
import sys
import pickle
import numpy as np
import numba as nb
from numba import types
from numba.extending import overload_method
from tqdm import tqdm


def parse_args():
    ''' parse command line arguments '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='verbose output', action='store_true')
    parser.add_argument('-p', '--parallel', help='parallel run', action='store_true')
    parser.add_argument('-nw', '--workers', help='job worker count', type=int, default=1)
    parser.add_argument('-nt', '--threads', help='threads per worker', type=int, default=8)
    args = parser.parse_args()
    return args.verbose, args.parallel, args.workers, args.threads


def client_info():
    ''' print client info '''
    info = str(CLIENT.scheduler_info)
    info = info.replace('<', '').replace('>', '').split()[6:8]
    print('\n%s %s' % tuple(info))


@nb.njit
def roll(a, sh):
    ''' numba (no python) implementation of np.roll '''
    n = a.size
    sh %= n
    ind = np.concatenate((np.arange(n-sh, n), np.arange(n-sh)))
    res = a.take(ind).reshape(a.shape)
    return res


@nb.njit
def correlation(t, dat, corr):
    ''' calculates correlation for sample '''
    # loop through time values
    for k in range(0, t.size):
        # standard time autocorrelation at time j from time i
        corr[k] = np.mean(np.multiply(dat, roll(dat, t[k]))+np.multiply(dat, roll(dat, -t[k])))/2
    # return correlation
    return corr

# main
if __name__ == '__main__':
    CWD = os.getcwd()
    VERBOSE, PARALLEL, NWORKER, NTHREAD = parse_args()
    # load data
    R = pickle.load(open(CWD+'/sga.r.pickle', 'rb'))
    ST = pickle.load(open(CWD+'/sga.spin.t.pickle', 'rb'))
    DAT = pickle.load(open(CWD+'/sga.spin.pickle', 'rb'))
    if VERBOSE:
        print('data loaded')
    # data shape
    NR, NS, NF = DAT.shape
    # time domain
    CT = ST-ST[np.int(ST.size/2)]
    pickle.dump(CT, open(CWD+'/sga.corr.t.pickle', 'wb'))
    if VERBOSE:
        print('correlation time domain dumped')
    # correlation array
    CORR = np.zeros((NR, NS, NF))
    if PARALLEL:
        os.environ['DASK_ALLOWED_FAILURES'] = '32'
        os.environ['DASK_MULTIPROCESSING_METHOD'] = 'forkserver'
        os.environ['DASK_LOG_FORMAT'] = '\r%(name)s - %(levelname)s - %(message)s'
        from multiprocessing import freeze_support
        from distributed import Client, LocalCluster, progress
        from dask import delayed
        # local cluster
        freeze_support()
        if NWORKER == 1:
            PROC = False
        else:
            PROC = True
        CLUSTER = LocalCluster(n_workers=NWORKER, threads_per_worker=NTHREAD, processes=PROC)
        # start client with local cluster
        CLIENT = Client(CLUSTER)
        # client information
        if VERBOSE:
            client_info()
        # submit futures to client for computation
        OPERS = [delayed(correlation)(CT, DAT[i, j], CORR[i, j]) for i in range(NR) for j in range(NS)]
        FUTURES = CLIENT.compute(OPERS)
        # progress bar
        if VERBOSE:
            progress(FUTURES)
        # gather results from workers
        RESULTS = CLIENT.gather(FUTURES)
        # assign correlations
        k = 0
        for i in range(NR):
            for j in range(NS):
                CORR[i, j, :] = RESULTS[k]
                k += 1
        # close client
        CLIENT.close()
    else:
        if VERBOSE:
            for i in tqdm(range(NR)):
                for j in tqdm(range(NS)):
                    CORR[i, j] = correlation(CT, DAT[i, j], CORR[i, j])
        else:
            for i in range(NR):
                for j in range(NS):
                    CORR[i, j] = correlation(CT, DAT[i, j], CORR[i, j])
    pickle.dump(CORR, open(CWD+'/sga.corr.pickle', 'wb'))
    if VERBOSE:
        print('\ncorrelations dumped')