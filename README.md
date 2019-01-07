# neuralAnderson


## Requirements


- NumPy
- SciPy
- scikit-learn
- MulticoreTSNE
- TensorFlow
- Keras
- Numba
- tqdm
- Dask
- Matplotlib


## Introduction


This project consists of a set of scripts used for predicting the transition between the strong-coupling (SC) and local moment (LM) phases of the soft-gap Anderson model using machine learning methods.


## Raw Data


The raw data consists of valid configurations for the 1-d soft-gap Anderson model for a given temperature and r-value parameterizing the density of states. The goal is to collect a large number of samples for a set of r-values at a fixed temperature very close to zero since the transition between the SC and LM phases occurs at zero temperature. Samples are generated using the Hirsch-Fye quantum Monte Carlo method.


## Parser


The parser script was written to handle the specific data output format from code used within our research group and as such, it will not necessarily be useful to the typical user. However, writing a parser for custom data formats is not too difficult. The parser outputs pickled python objects that comprise all of the necessary information for further calculation. This includes the r-values used to parameterize the samples from the density of states as well as the time domain for the feature samples. The r-values and the time domain are simply 1-d numpy arrays. The feature data is contained in a 3-d numpy array of shape (number of r-values, number of samples, number of features).


## Correlation


This script calculates the correlations of the parsed raw data. Both the time domain of the correlations as well as the correlations themselves are dumped as pickles.


## Machine Learning


This script uses machine learning methods to predict the transition between the SC and LM phases. The method involves multiple steps. The r-value domain as well as the associated raw data and correlations with their time domains are all loaded into memory from file. The method can be described as supervised learning aided by unsupervised learning. Thus, the unsupervised approach is treated first.


A fast Fourier transformation (FFT) is performed on the correlation data and the constant signal is filtered out to prevent the saturation values of the correlations from dominating the learning of the features. The FFT data is then scale, using a hyperbolic tangent scaler by default though others are available. The data is then transformed into a new orthogonal basis using principal component analysis (PCA), allowing direct inspection of projections that are associated with the greatest variance in the data. Following this, a nonlinear embedding may be optionally calculated, with the default being t-distributed stochastic neighbor embedding (t-SNE). In each case, the chosen number of projections is used to reduce the original feature space into a smaller number, reducing the dimensionality of the data. Only two or three projections are recommended. Samples in the reduced feature embedded space are then clustered into multiple classes using the spectral method by default. At least three clusters are recommended.


The raw data is scaled just as the correlation data was. A 1-d convolutional neural network (CNN) classifier is then fit to the scaled data. A CNN was chosen in order to pick up localized information in the raw data that may be useful for classifying a sample as belinging to either the SC or LM phase. The training sets are chosen according to the clusters obtained from the correlations according to the minimum and maximm average r-values, essentially picking out the most similar low and high energy samples from the data set. Many different parameters are available to customize the neural network. After the CNN is trained on the clusters predicted using unsupervised learning each sample is classified and assigned a probability of falling into each class corresponding to the SC and LM phases. For each set of samples corresponding to the same r-value, an average classification probability into the local moment phase alongside the standard deviation which is used to represent the fitting error. A sigmoid curve is fit to the classification probabilities dependent on the r-values produced from this. The midpoint of this sigmoid curve corresponds to the transition point, as this is the r-value where SC and LM samples are equally likely to be found.