# Copyright (c) 2021 Alexander E. Siemenn, Iddo Drori, Matthew J. Beveridge
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of
# conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or other materials provided with the
# distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import pandas as pd
import os
import GPy
import GPyOpt # Developed using GPyOpt version 1.2.6
from GPyOpt.methods import BayesianOptimization

def BO_optimizer(data, batch_size, param_path, save):
    '''
    Implement Bayesian optimization.

    Inputs:
    data          := adjusted parameter csv file (M by N+4) where N is the number of parameters and
                     M are the images with the computed total losses. Parameter values must be normalized;
                     the user may select whatever parameters they wish to use to tune to optimize droplets.
                     Losses are computed using the computer vision-based loss score in loss.py
    batch_size    := desired output batch size B for the suggested next locations
    save          := True or False value; saves predicted parameters for B samples as csv at the specified "param_path"

    Ouputs:
    df            := A dataframe of predicted, normalized parameter values (B by N), where N are the control parameters and B is the batch size
    '''
    X = np.array(data.iloc[:, 2:-2])  # X data are the parameters
    Y = np.array(data.iloc[:, -1])  # Y data is the last column
    N = X.shape[1]  # number of parameters, N
    bds = [{'name': f'x{n + 1}', 'type': 'continuous', 'domain': (0, 1)} for n in range(N)]  # N-dimensions

    kernel = GPy.kern.Matern52(input_dim=len(bds),
                               ARD=True)  # Use the matern 5/2 kernel with automatic relevence detection enabled
    optimizer = BayesianOptimization(f=None,
                                     domain=bds,
                                     constraints=None,
                                     model_type='GP',  # gaussian process model
                                     acquisition_type='EI',  # expected improvement acquisition
                                     acquisition_jitter=0.01,  # tune to adjust exploration
                                     X=X,  # normalized parameter value data
                                     Y=Y.reshape(Y.shape[0], 1),  # total loss data
                                     evaluator_type='local_penalization',
                                     batch_size=batch_size,  # batch size of predicted optima
                                     normalize_Y=False,
                                     kernel=kernel  # select the kernel
                                     )
    predicted = optimizer.suggest_next_locations()  # get next parameter values to synthesize experimentally
    names = np.array([f'predicted_{p + 1}' for p in range(predicted.shape[0])])
    names = names.reshape(names.shape[0], 1)
    df = pd.DataFrame(np.concatenate((names, predicted), axis=1),
                      columns=['Prediction'] + [f'Param{n + 1}' for n in range(N)])
    if save:
        df.to_csv(os.path.dirname(param_path) + '/bo_predicted_params.csv', sep=',', index=False)
    return df
