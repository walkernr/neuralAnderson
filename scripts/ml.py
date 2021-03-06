# -*- coding: utf-8 -*-
"""
Created on Mon Jul 02 22:25:48 2018

@author: Nicholas
"""

import argparse
import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from TanhScaler import TanhScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from scipy.odr import ODR, Model, RealData
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, AveragePooling1D, GlobalAveragePooling1D, Dropout, Dense
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import History


# parse command line
PARSER = argparse.ArgumentParser()
PARSER.add_argument('-v', '--verbose', help='verbose output', action='store_true')
PARSER.add_argument('-pt', '--plot', help='plot results', action='store_true')
PARSER.add_argument('-p', '--parallel', help='parallel run', action='store_true')
PARSER.add_argument('-nt', '--threads', help='number of threads',
                    type=int, default=16)
PARSER.add_argument('-mi', '--maxind', help='maximum set index',
                    type=int, default=25)
PARSER.add_argument('-sc', '--scaler', help='feature scaler',
                    type=str, default='tanh')
PARSER.add_argument('-rd', '--reduction', help='supervised dimension reduction method',
                    type=str, default='tsne')
PARSER.add_argument('-np', '--projections', help='number of embedding projections',
                    type=int, default=2)
PARSER.add_argument('-cl', '--clustering', help='clustering method',
                    type=str, default='spectral')
PARSER.add_argument('-nc', '--clusters', help='number of clusters',
                    type=int, default=4)
# PARSER.add_argument('-bk', '--backend', help='keras backend',
                    # type=str, default='tensorflow')
PARSER.add_argument('-ep', '--epochs', help='number of epochs',
                    type=int, default=16)
PARSER.add_argument('-lr', '--learning_rate', help='learning rate for neural network',
                    type=float, default=1e-3)

# parse arguments
ARGS = PARSER.parse_args()
# run specifications
VERBOSE = ARGS.verbose
PLOT = ARGS.plot
PARALLEL = ARGS.parallel
THREADS = ARGS.threads
MI = ARGS.maxind
SCLR = ARGS.scaler
RDCN = ARGS.reduction
NP = ARGS.projections
CLST = ARGS.clustering
NC = ARGS.clusters
# BACKEND = ARGS.backend
EP = ARGS.epochs
LR = ARGS.learning_rate

# random seed
SEED = 256
np.random.seed(SEED)
tf.random.set_seed(SEED)
# environment variables
# os.environ['KERAS_BACKEND'] = BACKEND
# if BACKEND == 'tensorflow':
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # from tensorflow import set_random_seed
    # set_random_seed(SEED)
# if PARALLEL:
    # os.environ['MKL_NUM_THREADS'] = str(THREADS)
    # os.environ['GOTO_NUM_THREADS'] = str(THREADS)
    # os.environ['OMP_NUM_THREADS'] = str(THREADS)
    # os.environ['openmp'] = 'True'
if not PARALLEL:
            THREADS = 1
DEVICE = '/CPU:0'
tf.config.threading.set_intra_op_parallelism_threads(THREADS)
tf.config.threading.set_inter_op_parallelism_threads(THREADS)
tf.device(DEVICE)
K.clear_session()

if PLOT:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    plt.rc('font', family='sans-serif')
    FTSZ = 48
    PPARAMS = {'figure.figsize': (26, 20),
               'lines.linewidth': 4.0,
               'legend.fontsize': FTSZ,
               'axes.labelsize': FTSZ,
               'axes.titlesize': FTSZ,
               'axes.linewidth': 2.0,
               'xtick.labelsize': FTSZ,
               'xtick.major.size': 20,
               'xtick.major.width': 2.0,
               'ytick.labelsize': FTSZ,
               'ytick.major.size': 20,
               'ytick.major.width': 2.0,
               'font.size': FTSZ}
    plt.rcParams.update(PPARAMS)

# information
if VERBOSE:
    print(66*'-')
    print('data loaded')
    print(66*'-')
    print('input summary')
    print(66*'-')
    print('plot:                      %d' % PLOT)
    print('parallel:                  %d' % PARALLEL)
    print('threads:                   %d' % THREADS)
    print('max index:                 %d' % MI)
    print('scaler:                    %s' % SCLR)
    print('reduction:                 %s' % RDCN)
    print('projections:               %d' % NP)
    print('clustering:                %s' % CLST)
    print('clusters:                  %d' % NC)
    # print('backend:                   %s' % BACKEND)
    print('network:                   %s' % 'cnn1d')
    print('epochs:                    %d' % EP)
    print('learning rate:             %.2e' % LR)
    print('fitting function:          %s' % 'logistic')
    print(66*'-')

CWD = os.getcwd()
OUTPREF = CWD+'/sga.%d.%s.%s.%d.%s.%d.cnn1d.%d.%.0e.logistic' \
          % (MI, SCLR, RDCN, NP, CLST, NC, EP, LR)
with open(OUTPREF+'.out', 'w') as out:
    out.write('# ' + 66*'-' + '\n')
    out.write('# input summary\n')
    out.write('# ' + 66*'-' + '\n')
    out.write('# plot:                      %d\n' % PLOT)
    out.write('# parallel:                  %d\n' % PARALLEL)
    out.write('# threads:                   %d\n' % THREADS)
    out.write('# max index:                 %d\n' % MI)
    out.write('# scaler:                    %s\n' % SCLR)
    out.write('# reduction:                 %s\n' % RDCN)
    out.write('# projections:               %d\n' % NP)
    out.write('# clustering:                %s\n' % CLST)
    out.write('# clusters:                  %d\n' % NC)
    # out.write('# backend:                   %s\n' % BACKEND)
    out.write('# network:                   %s\n' % 'cnn1d')
    out.write('# epochs:                    %d\n' % EP)
    out.write('# learning rate:             %.2e\n' % LR)
    out.write('# fitting function:          %s\n' % 'logistic')

EPS = 0.025 # np.finfo(np.float32).eps
# load data
R = np.load(CWD+'/sga.dat.npy')[:MI]
UT = np.load(CWD+'/sga.cor.t.npy')
ST = np.load(CWD+'/sga.dmp.t.npy')
UDAT = np.load(CWD+'/sga.cor.npy')[:MI]
SDAT = np.load(CWD+'/sga.dmp.npy')[:MI].astype(np.float32)
# data shape
UND, UNS, UNF = UDAT.shape
SND, SNS, SNF = SDAT.shape
# interactions
RS = np.concatenate(tuple(R[i]*np.ones(UNS) for i in range(UND)), 0)

if PLOT:
    CM = plt.get_cmap('plasma')
    SCALE = lambda r: (r-np.min(R))/np.max(R-np.min(R))
    if VERBOSE:
        print('colormap and scale initialized')
        print(66*'-')


# fitting function
def logistic(beta, t):
    ''' returns logistic sigmoid '''
    a = 0.0
    k = 1.0
    b, m = beta
    return a+np.divide(k, 1+np.exp(-b*(t-m)))


# odr fitting
def odr_fit(mpred, spred):
    ''' performs orthogonal distance regression '''
    dat = RealData(R, mpred, EPS*np.ones(len(R)), spred+EPS)
    mod = Model(logistic)
    odr = ODR(dat, mod, FITG)
    odr.set_job(fit_type=0)
    fit = odr.run()
    popt = fit.beta
    perr = fit.sd_beta
    trans = popt[1]
    cerr = perr[1]
    ndom = 256
    fdom = np.linspace(np.min(R), np.max(R), ndom)
    fval = logistic(popt, fdom)
    return trans, cerr, fdom, fval

if VERBOSE:
    print('fitting function initialized')
    print(66*'-')

# scaler dictionary
SCLRS = {'minmax':MinMaxScaler(feature_range=(0, 1)),
         'standard':StandardScaler(),
         'robust':RobustScaler(),
         'tanh':TanhScaler()}
# reduction dictionary
RDCNS = {'pca':PCA(n_components=0.99),
         'kpca':KernelPCA(n_components=NP, n_jobs=THREADS),
         'isomap':Isomap(n_components=NP, n_jobs=THREADS),
         'lle':LocallyLinearEmbedding(n_components=NP, n_jobs=THREADS),
         'tsne':TSNE(n_components=NP, perplexity=UNS,
                     early_exaggeration=12, learning_rate=200, n_iter=2000,
                     verbose=True, n_jobs=THREADS)}

if VERBOSE:
    print('scaling and reduction initialized')
    print(66*'-')

# neural network construction
def build_keras_cnn1d():
    ''' builds 1-d convolutional neural network '''
    model = Sequential([Conv1D(filters=int(SNF), kernel_size=4, activation='relu',
                               kernel_initializer='he_normal',
                               padding='valid', strides=1, input_shape=(SNF, 1)),
                        Conv1D(filters=int(SNF), kernel_size=4, activation='relu',
                               kernel_initializer='he_normal',
                               padding='valid', strides=1),
                        AveragePooling1D(strides=4),
                        Dropout(rate=0.25),
                        Conv1D(filters=int(SNF/4), kernel_size=4, activation='relu',
                               kernel_initializer='he_normal',
                               padding='valid', strides=1),
                        Conv1D(filters=int(SNF/4), kernel_size=4, activation='relu',
                               kernel_initializer='he_normal',
                               padding='valid', strides=1),
                        GlobalAveragePooling1D(),
                        Dropout(rate=0.5),
                        Dense(units=1, activation='sigmoid')])
    nadam = Nadam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.compile(loss='binary_crossentropy', optimizer=nadam, metrics=['mae', 'acc'])
    return model

NN = KerasClassifier(build_keras_cnn1d, epochs=EP, batch_size=32,
                     shuffle=True, verbose=VERBOSE, callbacks=[History()])

# clustering dictionary
CLSTS = {'agglomerative': AgglomerativeClustering(n_clusters=NC),
         'kmeans': KMeans(n_jobs=THREADS, n_clusters=NC, init='k-means++'),
         'spectral': SpectralClustering(n_jobs=THREADS, n_clusters=NC)}

if VERBOSE:
    print('neural network and clustering initialized')
    print(66*'-')

# scale unsupervised data
try:
    FUDOM = np.load(CWD+'/sga.%d.%s.fudat.k.npy' % (MI, SCLR))
    FUNF = FUDOM.size
    FUDAT = np.load(CWD+'/sga.%d.%s.fudat.npy' % (MI, SCLR)).reshape(UND*UNS, FUNF)
    FUDAT = np.concatenate((np.real(FUDAT), np.imag(FUDAT)), axis=1)
    SUDAT = np.load(CWD+'/sga.%d.%s.sudat.npy' % (MI, SCLR)).reshape(UND*UNS, 2*FUNF)
    if VERBOSE:
        print('scaled unsupervised data loaded from file')
except:
    FUDOM = np.fft.rfftfreq(UNF, 1)[1:]
    FUNF = FUDOM.size
    FUDAT = np.fft.rfft(UDAT.reshape(UND*UNS, UNF))[:, 1:]
    np.save(CWD+'/sga.%d.%s.fudat.k.npy' % (MI, SCLR), FUDOM)
    np.save(CWD+'/sga.%d.%s.fudat.npy' % (MI, SCLR), FUDAT.reshape(UND, UNS, FUNF))
    FUDAT = np.concatenate((np.real(FUDAT), np.imag(FUDAT)), axis=1)
    SUDAT = SCLRS[SCLR].fit_transform(FUDAT)
    np.save(CWD+'/sga.%d.%s.sudat.npy' % (MI, SCLR), SUDAT.reshape(UND, UNS, 2*FUNF))
    if VERBOSE:
        print('unsupervised data scaled')
if VERBOSE:
    print(66*'-')

# pca reduce unsupervised data
try:
    EVAR = np.load(CWD+'/sga.%d.%s.evar.npy' % (MI, SCLR))
    PUDAT = np.load(CWD+'/sga.%d.%s.pudat.npy' % (MI, SCLR)).reshape(UND*UNS, len(EVAR))
    PCOMP = np.load(CWD+'/sga.%d.%s.pcomp.npy' % (MI, SCLR))
    if VERBOSE:
        print('pca reduced unsupervised data loaded from file')
except:
    PUDAT = RDCNS['pca'].fit_transform(SUDAT)
    EVAR = RDCNS['pca'].explained_variance_ratio_
    PCOMP = RDCNS['pca'].components_
    np.save(CWD+'/sga.%d.%s.pudat.npy' % (MI, SCLR), PUDAT.reshape(UND, UNS, len(EVAR)))
    np.save(CWD+'/sga.%d.%s.evar.npy' % (MI, SCLR), EVAR)
    np.save(CWD+'/sga.%d.%s.pcomp.npy' % (MI, SCLR), PCOMP)
    if VERBOSE:
        print('unsupervised data pca reduced')
if VERBOSE:
    print(66*'-')
    print('principal components:     %d' % len(EVAR))
    print('explained variances:      %0.4f %0.4f %0.4f ...' % tuple(EVAR[:3]))
    print('total explained variance: %0.4f' % np.sum(EVAR))
    print(66*'-')

with open(OUTPREF+'.out', 'a') as out:
    out.write('# ' + 66*'-' + '\n')
    out.write('# pca fit\n')
    out.write('# ' + 66*'-' + '\n')
    out.write('# principal components:     %d\n' % len(EVAR))
    out.write('# explained variances:      %0.4f %0.4f %0.4f ...\n' % tuple(EVAR[:3]))
    out.write('# total explained variance: %0.4f\n' % np.sum(EVAR))

# reduction of unsupervised data
try:
    RUDAT = np.load(CWD+'/sga.%d.%s.%s.%d.rudat.npy' % (MI, SCLR, RDCN, NP)).reshape(UND*UNS, NP)
    if VERBOSE:
        print('nonlinearly reduced unsupervised data loaded from file')
except:
    if RDCN not in ('none', 'pca'):
        RUDAT = RDCNS[RDCN].fit_transform(PUDAT)
        np.save(CWD+'/sga.%d.%s.%s.%d.rudat.npy' % (MI, SCLR, RDCN, NP), RUDAT.reshape(UND, UNS, NP))
        if RDCN == 'tsne' and VERBOSE:
            print(66*'-')
        if VERBOSE:
            print('unsupervised data nonlinearly reduced')
    else:
        RUDAT = PUDAT[:, :NP]
    _, RUNF = RUDAT.shape

# clustering
if VERBOSE:
    print(np.max([66, 10+8*NC])*'-')
try:
    UPRED = np.load(CWD+'/sga.%d.%s.%s.%d.%s.%d.upred.npy' % (MI, SCLR, RDCN, NP, CLST, NC)).reshape(UND*UNS)
    if VERBOSE:
        print('clustered unsupervised data loaded from file')
except:
    UPRED = CLSTS[CLST].fit_predict(RUDAT)
    UCM = [np.mean(RS[UPRED == i]) for i in range(NC)]
    IUCM = np.argsort(UCM)
    for i in range(NC):
        UPRED[UPRED == IUCM[i]] = i+NC
    UPRED -= NC
    np.save(CWD+'/sga.%d.%s.%s.%d.%s.%d.upred.npy' % (MI, SCLR, RDCN, NP, CLST, NC), UPRED.reshape(UND, UNS))
    if VERBOSE:
        print('unsupervised data clustered')
UCM = [np.mean(RS[UPRED == i]) for i in range(NC)]
UCS = [np.std(RS[UPRED == i]) for i in range(NC)]
CUPRED = np.array([np.histogram(UPRED.reshape(UND, UNS)[i],
                                np.arange(NC+1))[0] for i in range(UND)])
UTRANS = R[np.argmin(np.std(CUPRED, 1))]
if VERBOSE:
    print(np.max([66, 10+8*NC])*'-')
    print('r\t' + NC*'c%d\t' % tuple(np.arange(NC)) + 'd')
    print(np.max([66, 10+8*NC])*'-')
    for i in range(UND):
        print('%0.2f\t' % R[i] + NC*'%04d\t' % tuple(CUPRED[i])+'%d' % np.argmax(CUPRED[i]))
    print(np.max([66, 10+8*NC])*'-')
    print('tot\t' + NC*'%04d\t' % tuple(np.sum(CUPRED, 0)))
    print('ave\t' + NC*'%0.2f\t' % tuple(UCM))
    print('std\t' + NC*'%0.2f\t' % tuple(UCS))
    print('trans\t%0.2f' % UTRANS)

with open(OUTPREF+'.out', 'a') as out:
    out.write('# ' + np.max([66, 10+8*NC])*'-' + '\n')
    out.write('# unsupervised learning results\n')
    out.write('# ' + np.max([66, 10+8*NC])*'-' + '\n')
    out.write('# r\t' + NC*'c%d\t' % tuple(np.arange(NC)) + 'd\n')
    out.write('# ' + np.max([66, 10+8*NC])*'-' + '\n')
    for i in range(UND):
        out.write('  %0.2f\t' % R[i] + \
                  NC*'%04d\t' % tuple(CUPRED[i]) + \
                  '%d\n' % np.argmax(CUPRED[i]))
    out.write('# ' + np.max([66, 10+8*NC])*'-' + '\n')
    out.write('# totals\n')
    out.write('# ' + 66*'-'+'\n')
    out.write('  ' + NC*'%04d\t' % tuple(np.sum(CUPRED, 0)) + '\n')
    out.write('# ' + 66*'-'+'\n')
    out.write('# averages\n')
    out.write('# ' + 66*'-'+'\n')
    out.write('  ' + NC*'%0.4f\t' % tuple(UCM) + '\n')
    out.write('# ' + 66*'-'+'\n')
    out.write('# standard deviations\n')
    out.write('# ' + 66*'-'+'\n')
    out.write('  ' + NC*'%0.4f\t' % tuple(UCS) + '\n')
    out.write('# ' + 66*'-'+'\n')
    out.write('# transition\n')
    out.write('# ' + 66*'-'+'\n')
    out.write('  %0.4f\n' % UTRANS)

# scale supervised data
try:
    SSDAT = np.load(CWD+'/sga.%d.%s.ssdat.npy' % (MI, SCLR)).reshape(SND*SNS, SNF)
    if VERBOSE:
        print(66*'-')
        print('scaled supervised data loaded from file')
except:
    SCLRS[SCLR].fit(np.real(SDAT.reshape(SND*SNS, SNF)[(UPRED == 0) | (UPRED == NC-1)]))
    SSDAT = SCLRS[SCLR].transform(SDAT.reshape(SND*SNS, SNF))
    np.save(CWD+'/sga.%d.%s.ssdat.npy' % (MI, SCLR), SSDAT.reshape(SND, SNS, SNF))
    if VERBOSE:
        print(125*'-')
        print('supervised data scaled')

# fit neural network to training data and predict classification data
try:
    LOSS = np.load(CWD+'/sga.%d.%s.%s.%d.%s.%d.cnn1d.%d.%.0e.loss.npy' % (MI, SCLR, RDCN, NP, CLST, NC, EP, LR))
    MAE = np.load(CWD+'/sga.%d.%s.%s.%d.%s.%d.cnn1d.%d.%.0e.mae.npy' % (MI, SCLR, RDCN, NP, CLST, NC, EP, LR))
    ACC = np.load(CWD+'/sga.%d.%s.%s.%d.%s.%d.cnn1d.%d.%.0e.acc.npy' % (MI, SCLR, RDCN, NP, CLST, NC, EP, LR))
    SPROB = np.load(CWD+'/sga.%d.%s.%s.%d.%s.%d.cnn1d.%d.%.0e.sprob.npy' % (MI, SCLR, RDCN, NP, CLST, NC, EP, LR))
    if VERBOSE:
        print(66*'-')
        print('neural network fit loaded from file')
except:
    if VERBOSE:
        print(125*'-')
    # fit training data
    LBLS = np.concatenate((np.zeros(np.sum(CUPRED[:, 0]), dtype=np.uint16),
                        np.ones(np.sum(CUPRED[:, NC-1]), dtype=np.uint16)), 0)
    NN.fit(SSDAT[(UPRED == 0) | (UPRED == NC-1), :, np.newaxis], LBLS)
    LOSS = NN.model.history.history['loss']
    MAE = NN.model.history.history['mae']
    ACC = NN.model.history.history['acc']
    np.save(CWD+'/sga.%d.%s.%s.%d.%s.%d.cnn1d.%d.%.0e.loss.npy' % (MI, SCLR, RDCN, NP, CLST, NC, EP, LR), LOSS)
    np.save(CWD+'/sga.%d.%s.%s.%d.%s.%d.cnn1d.%d.%.0e.mae.npy' % (MI, SCLR, RDCN, NP, CLST, NC, EP, LR), MAE)
    np.save(CWD+'/sga.%d.%s.%s.%d.%s.%d.cnn1d.%d.%.0e.acc.npy' % (MI, SCLR, RDCN, NP, CLST, NC, EP, LR), ACC)
    if VERBOSE:
        print(125*'-')
        print('neural network fitted to training data')
        print(66*'-')
    # predict classification data
    SPROB = NN.predict_proba(SSDAT[:, :, np.newaxis])[:, 1].reshape(SND, SNS)
    np.save(CWD+'/sga.%d.%s.%s.%d.%s.%d.cnn1d.%d.%.0e.sprob.npy' % (MI, SCLR, RDCN, NP, CLST, NC, EP, LR), SPROB)

MSPROB = np.mean(SPROB, 1)
SSPROB = np.std(SPROB, 1)
SPRED = SPROB.round()
SCM = [np.mean(RS[SPRED.reshape(-1) == i]) for i in range(2)]

# transition prediction
FITG = (1.0, UTRANS)
STRANS, SERR, SDOM, SVAL = odr_fit(MSPROB, SSPROB)
np.save(CWD+'/sga.%d.%s.%s.%d.%s.%d.cnn1d.%d.%.0e.strans.npy' % (MI, SCLR, RDCN, NP, CLST, NC, EP, LR), np.array([STRANS, SERR], dtype=np.float32))

if VERBOSE:
    print(66*'-')
    print('r\tave\tstd')
    print(66*'-')
    for i in range(SND):
        print('%0.2f\t' % R[i] + 2*'%0.2f\t' % (MSPROB[i], SSPROB[i]))
    print(66*'-')
    print('neural network predicted classification data')
    print(66*'-')
    print('trans\t'+2*'%0.2f\t' % (STRANS, SERR))
    print(66*'-')

with open(OUTPREF+'.out', 'a') as out:
    out.write('# ' + 66*'-' + '\n')
    out.write('# supervised learning results\n')
    out.write('# ' + 66*'-' + '\n')
    out.write('# epoch\tloss\tmae\tacc\n')
    out.write('# ' + 66*'-' + '\n')
    for i in range(EP):
        out.write('  %02d\t' % i + 3*'%0.4f\t' % (LOSS[i], MAE[i], ACC[i]) + '\n')
    out.write('# ' + 66*'-' + '\n')
    out.write('# r\tave\tstd\n')
    out.write('# ' + 66*'-' + '\n')
    for i in range(SND):
        out.write('  %0.4f\t' % R[i] + 2*'%0.4f\t' % (MSPROB[i], SSPROB[i]) + '\n')
    out.write('# ' + 66*'-' + '\n')
    out.write('# transition\n')
    out.write('# ' + 66*'-' + '\n')
    out.write('  '+2*'%0.4f\t' % (STRANS, SERR) + '\n')
    out.write('# ' + 66*'-'+'\n')

if PLOT:


    def plot_udat():
        ''' plot of domain averages for raw vdata '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        for j in range(UND):
            ax.plot(UT, np.mean(UDAT[j], 0), color=CM(SCALE(R[j])),
                    alpha=0.75, label=r'$r=%0.2f$' % R[j])
        for tick in ax.get_xticklabels():
            tick.set_rotation(16)
        scitxt = ax.yaxis.get_offset_text()
        scitxt.set_x(.025)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        ax.set_xlabel(r'$\mathrm{t}$')
        ax.set_ylabel(r'$\mathrm{Correlation}$')
        fig.savefig(OUTPREF+'.udat.png')


    def plot_rfudat():
        ''' plot of domain averages for imaginary fft data'''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        for j in range(UND):
            ax.plot(FUDOM, np.mean(FUDAT.reshape(UND, UNS, -1)[j][:, :FUNF], 0),
                    color=CM(SCALE(R[j])), alpha=0.75, label=r'$r=%0.2f$' % R[j])
        for tick in ax.get_xticklabels():
            tick.set_rotation(16)
        scitxt = ax.yaxis.get_offset_text()
        scitxt.set_x(.025)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        ax.set_xlabel(r'$\mathrm{k}$')
        ax.set_ylabel(r'$\mathrm{Real\enspace Correlation\enspace FFT}$')
        fig.savefig(OUTPREF+'.rfudat.png')


    def plot_ifudat():
        ''' plot of domain averages for real fft data'''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        for j in range(UND):
            ax.plot(FUDOM, np.mean(FUDAT.reshape(UND, UNS, -1)[j][:, FUNF:], 0),
                    color=CM(SCALE(R[j])), alpha=0.75, label=r'$r=%0.2f$' % R[j])
        for tick in ax.get_xticklabels():
            tick.set_rotation(16)
        scitxt = ax.yaxis.get_offset_text()
        scitxt.set_x(.025)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        ax.set_xlabel(r'$\mathrm{k}$')
        ax.set_ylabel(r'$\mathrm{Imaginary\enspace Correlation\enspace FFT}$')
        fig.savefig(OUTPREF+'.ifudat.png')


    def plot_rpca():
        ''' plot of real principal components '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        for j in range(NP+1):
            ax.plot(FUDOM, PCOMP[j][:FUNF], color=CM(0.15*(j+1)), alpha=0.75,
                    label=r'$\mathrm{principal\enspace component\enspace %d$' % j)
        for tick in ax.get_xticklabels():
            tick.set_rotation(16)
        scitxt = ax.yaxis.get_offset_text()
        scitxt.set_x(.025)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        ax.set_xlabel(r'$\mathrm{k}$')
        ax.set_ylabel(r'$\mathrm{Real\enspace Principal\enspace Component}$')
        fig.savefig(OUTPREF+'.rpca.png')


    def plot_ipca():
        ''' plot of real principal components '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        for j in range(NP+1):
            ax.plot(FUDOM, PCOMP[j][FUNF:], color=CM(0.15*(j+1)), alpha=0.75,
                    label=r'$\mathrm{principal\enspace component\enspace %d$' % j)
        for tick in ax.get_xticklabels():
            tick.set_rotation(16)
        scitxt = ax.yaxis.get_offset_text()
        scitxt.set_x(.025)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        ax.set_xlabel(r'$\mathrm{k}$')
        ax.set_ylabel(r'$\mathrm{Imaginary\enspace Principal\enspace Component}$')
        fig.savefig(OUTPREF+'.ipca.png')


    def plot_emb():
        ''' plot of reduced sample space '''
        fig = plt.figure()
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(1, 2),
                         axes_pad=2.0,
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="4%",
                         cbar_pad=0.4)
        for j in range(len(grid)):
            grid[j].spines['right'].set_visible(False)
            grid[j].spines['top'].set_visible(False)
            grid[j].xaxis.set_ticks_position('bottom')
            grid[j].yaxis.set_ticks_position('left')
        cbd = grid[0].scatter(RUDAT[:, 0], RUDAT[:, 1], c=RS, cmap=CM, s=120, alpha=0.05,
                              edgecolors='none')
        grid[0].set_aspect('equal', 'datalim')
        grid[0].set_xlabel(r'$x_0$')
        grid[0].set_ylabel(r'$x_1$')
        grid[0].set_title(r'$\mathrm{(a)\enspace Sample\enspace Embedding}$', y=1.02)
        for j in range(NC):
            grid[1].scatter(RUDAT[UPRED == j, 0], RUDAT[UPRED == j, 1],
                            c=np.array(CM(SCALE(UCM[j])))[np.newaxis, :], s=120, alpha=0.05,
                            edgecolors='none')
        grid[1].set_aspect('equal', 'datalim')
        grid[1].set_xlabel(r'$x_0$')
        grid[1].set_ylabel(r'$x_1$')
        grid[1].set_title(r'$\mathrm{(b)\enspace Cluster\enspace Embedding}$', y=1.02)
        cbar = grid[0].cax.colorbar(cbd)
        cbar.solids.set(alpha=1)
        grid[0].cax.toggle_label(True)
        fig.savefig(OUTPREF+'.emb.png')


    def plot_spred():
        ''' plot of prediction curves '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.plot(SDOM, SVAL, color=CM(SCALE(STRANS)),
                label=r'$\mathrm{Phase\enspace Probability\enspace Curve}$')
        ax.axvline(STRANS, color=CM(SCALE(STRANS)), alpha=0.50)
        for j in range(2):
            serrb = STRANS+(-1)**(j+1)*SERR
            ax.axvline(serrb, color=CM(SCALE(serrb)), alpha=0.50, linestyle='--')
        ax.scatter(R, MSPROB, color=CM(SCALE(R)), s=240, edgecolors='none', marker='*')
        ax.text(STRANS+np.diff(R)[0], .1,
                r'$r_{\mathrm{supervised}} = %.4f \pm %.4f$' % (STRANS, SERR))
        ax.set_ylim(0.0, 1.0)
        for tick in ax.get_xticklabels():
            tick.set_rotation(16)
        scitxt = ax.yaxis.get_offset_text()
        scitxt.set_x(.025)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        ax.set_xlabel(r'$\mathrm{r}$')
        ax.set_ylabel(r'$\mathrm{Probability}$')
        fig.savefig(OUTPREF+'.spred.png')

    def plot_semb():
        ''' plot of reduced sample space '''
        fig = plt.figure()
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(1, 2),
                         axes_pad=2.0,
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="4%",
                         cbar_pad=0.4)
        for j in range(len(grid)):
            grid[j].spines['right'].set_visible(False)
            grid[j].spines['top'].set_visible(False)
            grid[j].xaxis.set_ticks_position('bottom')
            grid[j].yaxis.set_ticks_position('left')
        cbd = grid[0].scatter(RUDAT[:, 0], RUDAT[:, 1], c=RS, cmap=CM, s=120, alpha=0.05,
                              edgecolors='none')
        grid[0].set_aspect('equal', 'datalim')
        grid[0].set_xlabel(r'$x_0$')
        grid[0].set_ylabel(r'$x_1$')
        grid[0].set_title(r'$\mathrm{(a)\enspace Sample\enspace Embedding}$', y=1.02)
        for j in range(2):
            grid[1].scatter(RUDAT[SPRED.reshape(-1) == j, 0], RUDAT[SPRED.reshape(-1) == j, 1],
                            c=np.array(CM(SCALE(SCM[j])))[np.newaxis, :], s=120, alpha=0.05,
                            edgecolors='none')
        grid[1].set_aspect('equal', 'datalim')
        grid[1].set_xlabel(r'$x_0$')
        grid[1].set_ylabel(r'$x_1$')
        grid[1].set_title(r'$\mathrm{(b)\enspace Classification\enspace Embedding}$', y=1.02)
        cbar = grid[0].cax.colorbar(cbd)
        cbar.solids.set(alpha=1)
        grid[0].cax.toggle_label(True)
        fig.savefig(OUTPREF+'.semb.png')

    plot_udat()
    plot_rfudat()
    plot_ifudat()
    plot_rpca()
    plot_ipca()
    plot_emb()
    plot_spred()
    plot_semb()

    if VERBOSE:
        print('plots saved')
        print(66*'-')
