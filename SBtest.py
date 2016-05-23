from matplotlib import pyplot as plt
from numpy import random as rd
from scipy import io as sio
import SparseBayesian as sb
import numpy as np
import time

def regression(noise, basis, targets, test_basis, test_targets):
    options = sb.OptionSet(max_iter=500, diagnosticLevel=2, monitor=10)
    settings = sb.ParamSet(noiseStdDev=0.1)
    t_start = time.time()
    Param, HyperParam, D = sb.SparseBayesian('gaussian', basis, targets, options, settings)
    t_end = time.time()
    N,M = basis.shape
    w_infer = np.zeros((M,1))
    w_inder[Param.relevant] = Param.value
    y = test_basis*w_infer
    error = np.mean(np.power(test_targets-y,2))
    w_num = len(Param.relevant)

    f_Rows = 1
    f_Cols = 2
    SP_Likely = 1
    SP_WEIGHTS = 2

    fig = plt.figure(1)
    ax = fig.add_subplot(f_Rows, f_Cols, SP_LIKELY)
    lsteps = D.likelihood.shape[0]
    plt.plot(range(1,lsteps+1), D.likelihood, 'g-')
    ax.set_xlim(0, lstep+1)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Log Marginal Likelihood')
    ax.title('Log Marginal Likelihood Trace',fontsize=12)
    lim = ax.axis()
    dx = lim[1]-lim[0]
    dy = lim[3]-lim[2]
    s = 'Actural Noise:    %.5f'%noise
    ax.text(lim[0]+0.1*dx,lim[2]+0.6*dy, s)
    s = 'Inferred Noise:   %.5f'%(1/np.sqrt(HyperParam.beta))
    ax.text(lim[0]+0.1*dx,lim[2]+0.5*dy, s)

    ax = fig.add_subplot(f_Rows, f_Cols, SP_WEIGHTS)
    ax.set_xlim(0, N+1)
    ax.set_ylabel('Inferred Weights')
    ax.stem(w_infer)
    ax.title('Inferred Weights (%d)'%len(Param.relevant),fontsize=12)
    return fig, error, w_num, t_end-t_start

def classification(X, basis, targets):
    options = sb.OptionSet(max_iter=500, diagnosticLevel=2, monitor=10)
    t_start = time.time()
    Param, HyperParam, D = SparseBayesian('Bernoulli', basis, targets, options)
    N,M = basis.shape
    w_infer = np.zeros((M,1))
    w_infer[Param.relevant] = Param.value
    y = np.divide(1,1+np.exp(-basis*w_infer))
    c = np.double(y>=0.5)
    mis = c[c!=targets].shape[1]/N
    t_end = time.time()

    f_Rows = 1
    f_Cols = 2
    SP_LIKELY = 1
    SP_SCATTER = 2
    sqrtN = sqrt(size(X,1))
    
    fig = plt.figure(1)
    ax = fig.add_subplot(f_Rows, f_Cols, SP_LIKELY)
    lsteps = D.likelihood.shape[0]
    ax.plot(range(1,lsteps+1), D.lieklihood, 'g-')
    ax.set_xlim(0, lsteps+1)
    ax.title('Log Marginal Likelihood Trace', fontsize=12)

    ax = subplot(f_Rows, f_Cols, SP_SCATTER)
    x_min = np.min(X[:,0])
    x_max = np.max(X[:,0])
    x_tick = (x_max-x_min)/sqrtN
    ax.set_xlim(x_min-x_tick, x_max+x_tick)
    y_min = np.min(X[:,1])
    y_max = np.max(X[:,1])
    y_tick = (y_max-y_min)/sqrtN
    ax.set_ylim(y_max-y_min)/sqrtN
    ax.scatter(X[:,0][targets==1],X[:,1][targets==1], 'r.')
    ax.scatter(X[:,0][targets==0],X[:,1][targets==0], 'b.')
    ax.scatter(X[:,0][c==1], X[:,1][c==1], 'ro')
    ax.scatter(X[:,0][c==0], X[:,1][c==0], 'bo')
    ax.title('Distribution of Classified Points', fontsize=12)

    return fig, mis, t_end-t_start

def run(datafile, it, batch):
    raw = np.matrix(sio.loadmat(datafile))
    error = np.zeros((it,1))
    w_num = np.zeros((it,1))
    time  = np.zeros((it,1))
    for i in range(it):
        N = batch*(i+1)
        basis = raw[range(N),0:-1]
        targets = raw[range(N),-1]
        noise = np.std(targets)*0.2
        test_basis = raw[N+1:,0:-1]
        test_targets = raw[N+1:,-1]
        fig1, error[i,0], w_num[i,0], time[i,0] = regression(noise, basis, targets, test_basis, test_targets)
    
    fig2 = plt.figure(2)
    ax = fig2.add_subplot(121)
    ax.plot(range(batch, batch*(it+1), batch), error)
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('MSE')
    ax.title('Regression Error')
    ax = fig2.add_subplot(122)
    ax.plot(range(batch, batch*(it+1), batch), w_num)
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Number of Relevant Vectors')
    ax.title('Number of Utilized Basis', fontszie=12)


