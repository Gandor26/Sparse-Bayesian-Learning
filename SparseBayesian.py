from os import _exit
import sys
import time
import numpy as np
import scipy as sp
import random as rd

def sigmoid(x):
    return 1/(1+np.exp(-x))

def PreProcessBasis(basis):
    N,M = np.shape(basis)
    scale = np.sqrt(np.sum(np.multiply(basis, basis), 0))
    scale[scale==0] = 1
    for m in range(M):
        basis[:,m] = basis[:,m]/scale[0,m]
    return basis, scale

Likelihoods = {'GAUSSIAN':1, 'BERNOULLI':2, 'POISSON':3}
def getLikelihood(likelihood_str):
    try:
        Likelihood = Likelihoods[likelihood_str.upper()]
    except KeyError:
        print('Unknown Likelihood Type')
        _exit(1)
    return Likelihood
    
def ParamSet(**argin):
    settings = {}
    settings['relevant'] = set()
    settings['mu'] = None
    settings['alpha'] = None
    for property_str in argin:
        upper_str = property_str.upper()
        if upper_str == 'BETA':
            settings['beta'] = argin[property_str]
        elif upper_str == 'NOISESTDDEV':
            settings['noiseStdDev'] = argin[property_str]
        elif upper_str == 'ALPHA':
            settings['alpha'] = argin[property_str]
        elif upper_str == 'WEIGHTS':
            settings['mu'] = argin[property_str]
        elif upper_str == 'RELEVANT':
            settings['relevant'] = set(argin[property_str])
        else:
            print('Unrecognized Parameter Type of "%s"' % property_str)
            _exit(1)
    return settings

def OptionSet(**argin):
    options = {}
    options['fixedNoise'] = False
    options['freeBasis'] = set()
    options['max_iter'] = 500
    options['max_time'] = 1000      #seconds
    options['monitor'] = 0
    options['diagnosticLevel'] = 3
    options['diagnosticFID'] = 1
    options['diagnosticFile'] = None
    
    def timeFormat(time):
        s = time.split(' ')
        if len(s) == 2:
            v = int(s[0])
            r = s[1].upper()
            if r in ('SECONDS', 'SECOND'):
                s = v
            elif r in ('MINUTES', 'MINUTE'):
                s = v*60
            elif r in ('HOURS', 'HOUR'):
                s = v*3600
            else:
                print('Badly formed time argument')
                _exit(1)
        return s

    for option_str in argin:
        upper_str = option_str.upper()
        if upper_str == 'FIXEDNOISE':
            options['fixedNoise'] = argin[option_str]
        elif upper_str == 'FREEBASIS':
            options['freeBasis'].update(argin[option_str])
        elif upper_str == 'ITERATIONS':
            options['max_iter'] = argin[option_str]
        elif upper_str == 'TIME':
            options['max_time'] = timeFormat(argin[option_str])
        elif upper_str == 'MONITOR':
            options['monitor'] = argin[option_str]
        elif upper_str == 'DIAGNOSTICLEVEL':
            if argin[option_str] >= 0 and argin[option_str] <=4:
                options['diagnosticLevel'] = argin[option_str]
            else:
                print('Illegal Assignment of Diagnostic Level: %d'%
                      argin[option_str])
                _exit(1)
        elif upper_str == 'DIAGNOSTICFILE':
            options.diagnosticFID = 0
            options.diagnosticFile = argin[option_str]
        else:
            print('Unrecognized Option Item')
            _exit(1)
    return options

def Diagnostics(options, level, msg = None, **argin):
    def isnum(value):
        try:
            value=value+1
        except TypeError:
            return False
        else:
            return True

    f = sys.stdout
    if isnum(level):
        if level <= options['diagnosticLevel']:
            print(msg, file=f)
    else:
        if level.upper() in set(['OPEN', 'START']):
            if options['diagnosticFID'] != 1:
                try:
                    f = open(options['diagnosticFile'], 'w')
                except FileNotFoundError:
                    print('Could not open diagnostic file %s' %\
                          options['diagnosticFile'])
                    _exit(1)
        elif level.upper() in set(['CLOSE', 'END', 'FINISH']):
            if options['diagnosticFID'] != 1:
                f.close()
    #return options

def Initialization(likelihood, basis, targets, settings, options):
    GAUSSIAN_SNR_INIT = 0.1
    INIT_ALPHA_MAX = 1e3
    INIT_ALPHA_MIN = 1e-3

    Basis, Scale = PreProcessBasis(basis)
    Likelihood = getLikelihood(likelihood)
    if Likelihood == Likelihoods['GAUSSIAN']:
        if 'beta' in settings:
            beta = settings['beta']
        elif 'noiseStdDev' in settings:
            beta = 1/(settings['noiseStdDev']**2)
        else:
            beta = 1/(np.max([1e-6, np.std(targets, ddof=1)])*GAUSSIAN_SNR_INIT) ** 2
    else:
        beta = np.array([])

    targetsPseudoLinear = targets
    if Likelihood == Likelihoods['BERNOULLI']:
        targetsPseudoLinear = targets*2-1
    elif Likelihood == Likelihoods['POISSON']:
        targetsPseudoLinear = np.log(targets+1e-3)

    extra = options['freeBasis'] - settings['relevant']
    used = settings['relevant'] | extra
    if not used:
        proj = Basis.T * targetsPseudoLinear
        used = np.argmax(np.abs(proj))
        Diagnostics(options, 2,\
            'Initializing with the maximally aligned basis vector(%d)'% used)
        used = set([used])
    else:
        Diagnostics(options, 2,\
            'Initializing with supplied vectors with size=%d'% len(used))
    Phi = Basis[:,list(used)]
    Mt = len(used)
    order = [item for item in used]


    if not settings['mu']:
        if Likelihood == Likelihoods['GAUSSIAN']:
            mu = np.array([], dtype=float)
        elif Likelihood == Likelihoods['BERNOULLI']:
            tmp = (targetsPseudoLinear*0.9+1)/2
            mu = np.linalg.lstsq(Phi, np.log(np.divide(tmp,1-tmp)))[0]
            tmp = np.log(np.divide(tmp,1-tmp))
        elif Likelihood == Likelihoods['POISSON']:
            mu = np.linalg.lstsq(Phi, targetsPseudoLinear)
    else:
        if len(settings['mu']) != len(settings['relevent']):
            print('Basis length (%d) should equal weight vector length (%d)'\
                %(len(settings['mu']), len(settings['relevant'])))
            _exit(1)
        Diagnostics(options, 2, 'Initializing with supplied weights')
        mu = np.vstack([settings['mu'], np.zeros((len(extra), 1))])

    if not settings['alpha']:
        if Likelihood == Likelihoods['GAUSSIAN']:
            p = np.diag(Phi.T*Phi)*beta
            q = (Phi.T*targets)*beta
            alpha = np.power(p,2)/(np.power(q,2)-p)
            if np.all(alpha<0):
                Diagnostics(options, 1,\
                    'Warning: no relevant basis function at initialization!')
            alpha[alpha<0] = INIT_ALPHA_MAX
        elif Likelihood == Likelihoods['BERNOULLI'] or\
             Likelihood == Likelihoods['POISSON']:
            alpha = 1/np.power(mu+np.array((mu==0),dtype=float), 2)
            alpha[alpha<INIT_ALPHA_MIN] = INIT_ALPHA_MIN
            alpha[alpha>INIT_ALPHA_MAX] = INIT_ALPHA_MAX
    else:
        if len(settings['alpha']) != len(settings['relevant']):
            print('Basis length (%d) should equal alpha vector length (%d)' %\
                  (len(settings['relevant']), len(settings['alpha'])))
            _exit(1)
        alpha = np.vstack(alpha, np.zeros((len(extra), 1)))

    alpha[list(options['freeBasis']-used)] = 1e-6
    
    return Likelihood, Basis, Scale, alpha, beta, mu, Phi, used, order

def ControlSet():
    controls = {}
    controls['ZeroFactor'] = 1e-12
    controls['MinDeltaLogAlpha'] = 1e-3
    controls['MinDeltaLogBeta'] = 1e-6
    controls['AdditionPriority'] = False
    controls['DeletionPriority'] = True
    controls['betaUpdateStart'] = 10
    controls['betaUpdateFrequency'] = 5
    controls['betaMaxFactor'] = 1e6
    controls['PosteriorModeFrequency'] = 1
    controls['BasisAlignmentTest'] = True
    controls['AlignmentMax'] = 1-1e-3

    return controls

def PosteriorMode(Likelihood, Basis, targets, alpha, mu, iterMax,\
        options):
    def DataError(Likelihood, Basis_mu, targets):
        if Likelihood == Likelihoods['BERNOULLI']:
            y = sigmoid(Basis_mu)
            if (y==0)[targets>0].any() or (y==1)[targets<1].any():
                e = float('inf')
            else:
                y0 = y!=0
                y1 = y!=1
                e = -(targets[y0].T*np.log(y[y0].reshape(-1,1))+\
                    (1-targets[y1]).T*np.log(1-y[y1].reshape(-1,1)))
        elif Likelihood == Likelihoods['POISSON']:
            y = np.exp(Basis_mu)
            e = -np.sum(np.multiply(targets, Basis_mu)-y)
        
        return e, y
    
    GRAD_MIN = 1e-6
    STEP_MIN = 1/(2**8)
    Mt = np.shape(Basis)[1]
    A = np.diag(alpha)
    Basis_Mu = Basis*mu
    dataError, y = DataError(Likelihood, Basis_Mu, targets)
    regularizer = (alpha.T*np.power(mu, 2))/2
    totalError = dataError + regularizer
    badHess = False
    errorLog = np.zeros((iterMax, 1))

    for it in range(iterMax):
        errorLog[it, 0] = totalError
        Diagnostics(options, 4, 'PosteriorMode Cycle:%2d\terror:%.6f'\
                %(it, totalError))
        e = targets-y
        g = Basis.T*e - np.multiply(alpha,mu)
        if Likelihood == Likelihoods['BERNOULLI']:
            beta = np.multiply(y,1-y)
        elif Likelihood == Likelihoods['POISSON']:
            beta = y
        Basis_b = np.multiply(Basis, beta*np.ones((1,Mt)))
        Hess = Basis_b.T*Basis + A
        try:
            U = np.linalg.cholesky(Hess)
        except np.linalg.linalg.LinAlgError:
            Diagnostics(options, 1, 'Warning: ill-conditioned Hessian')
            badHess = True
            U = np.matrix([])
            beta = np.array([])
            LikelihoodMode = np.array([])
            break
        if (np.abs(g)<GRAD_MIN).all():
            errorLog = errorLog[1:it, 0]
            Diagnostics(options, 4, 'PosteriorMode Convergence (<1e-6) after\
                    %d iterations'% it)
            break
        delta_mu = np.linalg.solve(L.T,np.linalg.solve(L, g))
        step = 1.0
        while step > STEP_MIN:
            mu_new = mu + step*delta_mu
            Basis_mu = Basis*mu_new
            [dataError, y] = DataError(Likelihood, Basis_mu, targets)
            regularizer = (alpha.T*np.power(mu, 2))/2
            totalError = dataError + regularizer
            if totalError >= errorLog[it, 1]:
                step = step/2
                Diagnostics(options, 4, 'PosteriorMode Error increase! Backing\
                        off to l = %.3f' % step)
            else:
                mu = mu_new
                step = 0.0

        if step>0:
            Diagnostics(options, 4, 'PosteriorMode stopping due to back-off\
                    limie (|g|=%.3f)'%np.max(np.abs(g)))
            break
    LikelihoodMode = -dataError

    return mu, U, beta, LikelihoodMode, badHess

def updateStats(Likelihood, Basis, Phi, targets, order, alpha, beta, mu,\
        Basis_Phi, Basis_targets, options):
    MAX_POSTMODE_ITER = 25
    N = np.shape(Basis)[0]
    Mt = np.shape(Phi)[1]
    if Likelihood == Likelihoods['GAUSSIAN']:
        U = np.linalg.cholesky(Phi.T*Phi*beta +\
                               np.diag(alpha.ravel().tolist()[0]))
        U_inv = U.I
        Sigma = U_inv.T*U_inv
        mu = beta*(Sigma*(Phi.T*targets))
        y = Phi*mu
        e = targets - y
        dataLikelihood = (N*np.log(beta)-beta*(e.T*e))/2
    else:
        mu, U, beta, dataLikelihood, badHess =\
            PosteriorMode(Likelihood, Phi, targets, alpha, mu, MAX_POSTMODE_ITER, options)
        U_inv = U.I
        Sigma = U_inv.T*U_inv
        if Likelihood == Likelihoods['BERNOULLI']:
            y = sigmoid(Phi*mu)
        elif Likelihood == Likelihoods['POISSON']:
            y = np.exp(Phi*mu)
        e = targets - y

    logML = dataLikelihood - np.power(mu, 2).T*alpha/2 +\
    np.sum(np.log(alpha))/2-np.sum(np.log(np.diag(U)))
    diagSigma = np.matrix(np.diag(Sigma)).reshape(-1,1)
    gamma = 1 - np.multiply(alpha, diagSigma)

    if Likelihood == Likelihoods['GAUSSIAN']:
        b_Basis_Phi = beta*Basis_Phi
        tmp = b_Basis_Phi*U_inv.T
        S_in = (beta - np.diag(tmp*tmp.T)).reshape(-1,1)
        Q_in = beta*(Basis_targets - Basis_Phi*mu)
    else:
        b_Basis_Phi = Basis.T * (np.multiply(Phi, beta*np.ones((1, Mt))))
        tmp = b_Basis_Phi*U_inv.T
        S_in = ((beta.T*np.power(Basis,2))-np.diag(tmp*tmp.T)).reshape(-1, 1)
        Q_in = Basis.T*e

    S_out = S_in.copy()
    Q_out = Q_in.copy()
    index = order
    tmp = alpha-S_in[index]
    #print(np.multiply(alpha,S_in[index]))
    S_out[index] = np.divide(np.multiply(alpha,S_in[index]), tmp)
    Q_out[index] = np.divide(np.multiply(alpha,Q_in[index]), tmp)
    factor = np.power(Q_out,2) - S_out

    return Sigma, mu, S_in, Q_in, S_out, Q_out, factor, logML, \
           gamma, b_Basis_Phi, beta

def Bayesian(likelihood, basis, targets, settings = None, options = None):
    if not settings:
        settings = ParamSet()
    if not options:
        options = OptionSet()
    if (options['fixedNoise']) and ('beta' not in settings)\
       and ('noiseStdDev' not in settings):
        print('Options to fix noise variance but no value supplied')
        _exit(1)

    Res = {}
    controls = ControlSet()
    t_start = time.time()

    Diagnostics(options, 'start')
    Likelihood, Basis, Scale, alpha, beta, mu, Phi, used, order = \
            Initialization(likelihood, basis, targets, settings, options)
    if Likelihood == Likelihoods['GAUSSIAN']:
        Basis_Phi = Basis.T*Phi
    else:
        Basis_Phi = np.matrix([])
    Basis_targets = Basis.T*targets

    Sigma, mu, S_in, Q_in, S_out, Q_out, factor, logML, gamma, b_Basis_Phi,\
            beta = updateStats(Likelihood, Basis, Phi, targets, order, alpha,\
            beta, mu, Basis_Phi, Basis_targets, options)
    if options['max_iter'] == 0:
        Param = np.array([])
        HyperParam = np.array([])
        Res['Likelihood'] = logML
        return Param, HyperParam, Res
    N,M_f = np.shape(Basis)
    Mt = np.shape(Phi)[1]
    add_count = delete_count = update_count = 0
    maxLogSize = options['max_iter']+controls['betaUpdateStart']+\
            options['max_iter']//controls['betaUpdateFrequency']
    logMarginalLog = np.zeros((maxLogSize, 1))
    count = 0

    if controls['BasisAlignmentTest']:
        aligned_in = np.array([]).reshape(-1,1)
        aligned_out = np.array([]).reshape(-1,1)
        alignDeferCount = 0

    ACTION = {'reestimate':0, 'add':1, 'delete':-1, 'terminate':2,\
            'noise_only':11, 'alignment_skip':12}
    
    '''         MAIN  LOOP          '''
    it = 0
    LAST_ITER = False
    while not LAST_ITER:
        it = it+1
        update_iter = Likelihood==Likelihoods['GAUSSIAN'] or\
                it%controls['PosteriorModeFrequency']
        deltaML = np.zeros((M_f, 1))
        Action = np.zeros((M_f, 1))
        
        '''     find vectors needing re-estimation       '''

        iu = [i for i in range(len(order)) if factor[order[i]]>controls['ZeroFactor']]
        index = [order[i] for i in iu]
        new_alpha = np.divide(np.power(S_out[index],2),factor[index])
        delta = (1/new_alpha - 1/alpha[iu])
        tmp1 = np.multiply(delta, np.power(Q_in[index],2))
        tmp2 = np.multiply(delta, S_in[index])+1
        deltaML[index] = (np.divide(tmp1,tmp2)-np.log(tmp2))/2

        '''     find vectors needing deletion       '''

        iu = [i for i in range(len(order)) if factor[order[i]]<=controls['ZeroFactor']]  
        index = [order[i] for i in iu]
        any_to_delete = Mt>1 and not (set(index)-options['freeBasis']).issubset({})
        if any_to_delete:
            tmp1 = np.divide(np.power(Q_out[index], 2), alpha[iu])
            tmp2 = np.divide(S_out[index],alpha[iu])-1
            deltaML[index] = np.divide(tmp1, tmp2)-np.log(-tmp2)
            Action[index] = ACTION['delete']

        '''     find vectors needing addition       '''
        index = set([i for i in range(M_f) if factor[i]>controls['ZeroFactor']])-used
        if controls['BasisAlignmentTest']:
            index -= set(aligned_out.ravel().tolist())
        index = list(index)
        any_to_add =  len(index) > 0
        if any_to_add:
            tmp = np.divide(np.power(Q_in[index],2), S_in[index])
            deltaML[index] = (tmp-1-np.log(tmp))/2
            Action[index] = ACTION['add']
        deltaML[list(options['freeBasis'])] = 0
        flag_add = any_to_add and controls['AdditionPriority']
        flag_delete = any_to_delete and controls['DeletionPriority']
        if flag_add or flag_delete:
            deltaML[Action==ACTION['reestimate']] = 0
            if flag_add and not controls['DeletionPriority']:
                deltaML[Action==ACTION['delete']] = 0
            if flag_delete and not controls['AdditionPriority']:
                deltaML[Action==ACTION['add']] = 0
        
        '''find the most significant update to decide next step'''

        deltaLogMarginal = np.max(deltaML)
        nu = np.argmax(deltaML)
        selectedAction = Action[nu]
        action_worth = deltaLogMarginal>0
        if selectedAction in [ACTION['reestimate'], ACTION['delete']]:
            try:
                j = order.index(nu)
            except ValueError:
                if not action_worth:
                    j = 0
                else:
                    print('Selected vector %d is not in basis but operated!'% nu)
                    _exit(1)
        phi = Basis[:, nu]
        new_alpha = S_out[nu]**2/factor[nu]

        if (not action_worth) or\
           (selectedAction == ACTION['reestimate'] and \
            np.abs(np.log(new_alpha/alpha[j]))<controls['MinDeltaLogAlpha']\
            and not any_to_delete):
            selectedAction = ACTION['terminate']
            act = 'Potential Termination'
        
        if controls['BasisAlignmentTest']:
            if selectedAction == ACTION['add']:
                p = phi.T*Phi
                aligned_pos = [order[i] for i in range(Mt) if p[0,i]>controls['AlignmentMax']]
                aligned_num = len(aligned_pos)
                if aligned_num > 0:
                    selectedAction = ACTION['alignment_skip']
                    act = 'alignment-deferred addition'
                    alignDeferCount = alignDeferCount+1
                    aligned_out = np.vstack((aligned_out,\
                            nu*np.ones((aligned_num,1))))
                    aligned_in = np.vstack((aligned_in,\
                            np.array(aligned_pos).reshape(-1,1)))
            elif selectedAction == ACTION['delete']:
                aligned_pos = (aligned_in == nu).nonzero()[0]
                aligned_num = len(aligned_pos)
                if aligned_num>0:
                    aligned_in = np.delete(aligned_in,aligned_pos,axis=0)
                    aligned_out = np.delete(aligned_out, aligned_pos, axis=0)
                    Diagnostics(options, 3, 'Alignment reinstated')

        update_required = False
        if selectedAction == ACTION['reestimate']:
            old_alpha = alpha[j].copy()
            alpha[j] = new_alpha
            s_j = Sigma[:,j]
            delta = 1/(new_alpha-old_alpha)
            kappa = 1/(Sigma[j,j]+delta)
            tmp = s_j*kappa
            new_Sigma = Sigma - tmp*s_j.T
            delta_mu = -mu[j,0]*tmp
            mu = mu + delta_mu
            if update_iter:
                S_in = S_in + np.power(b_Basis_Phi*s_j,2)*kappa
                Q_in = Q_in - b_Basis_Phi*delta_mu
            update_count = update_count+1
            act = 're-estimation'
            update_required = True
        elif selectedAction == ACTION['add']:
            if Likelihood == Likelihoods['GAUSSIAN']:
                Basis_phi = Basis.T*phi
                Basis_Phi = np.hstack((Basis_Phi, Basis_phi))
                beta_phi = beta*phi
                b_Basis_phi = beta*Basis_phi
            else:
                beta_phi = np.multiply(beta, phi)
                b_Basis_phi = Basis.T*beta_phi
            tmp = ((beta_phi.T*Phi)*Sigma).T
            alpha = np.vstack((alpha, new_alpha))
            Phi = np.hstack((Phi, phi))
            s_ii = 1/(new_alpha+S_in[nu,0])
            s_i = -tmp*s_ii
            tau = -s_i*tmp.T
            new_Sigma = np.vstack((np.hstack((Sigma+tau, s_i)),\
                    np.hstack((s_i.T, s_ii))))
            mu_i = s_ii*Q_in[nu,0]
            delta_mu = np.vstack((-tmp*mu_i, mu_i))
            mu = np.vstack((mu, 0)) + delta_mu
            if update_iter:
                mCi = b_Basis_phi - b_Basis_Phi*tmp
                S_in = S_in-np.power(mCi,2)*s_ii
                Q_in = Q_in - mCi*mu_i
            used |= {nu}
            order.append(nu)
            add_count = add_count+1
            act = 'addition'
            update_required = True
        elif selectedAction == ACTION['delete']:
            if Likelihood == Likelihoods['GAUSSIAN']:
                Basis_Phi = np.delete(Basis_Phi, j, axis=1)
            Phi = np.delete(Phi, j, axis=1)
            alpha = np.delete(alpha, j, axis=0)
            s_jj = Sigma[j,j].copy()
            s_j = Sigma[:,j].copy()
            tmp = s_j/s_jj
            new_Sigma = Sigma-tmp*s_j.T
            new_Sigma = np.delete(new_Sigma, j, axis=0)
            new_Sigma = np.delete(new_Sigma, j, axis=1)
            delte_mu = -mu[j,0]*tmp
            mu_j = mu[j,0].copy()
            mu = mu + delta_mu
            mu = np.delete(mu, j, axis=0)
            if update_iter:
                jPm = b_Basis_Phi * s_j
                S_in = S_in + np.power(jPm,2)/s_jj
                Q_in = Q_in + mu_j*jPm/s_jj
            used -= {nu}
            order.remove(nu)
            delete_count = delete_count+1
            act = 'deletion'
            update_required = True
        Mt = len(order)
        Diagnostics(options, 3, 'Action: %s of %d (%g)'\
                %(act, nu, deltaLogMarginal))
        
        '''     update main statistics      '''
        
        if update_required:
            if update_iter:
                S_out = S_in.copy()
                Q_out = Q_in.copy()
                tmp = np.divide(alpha, alpha-S_in[order])
                S_out[order] = np.multiply(tmp, S_in[order])
                Q_out[order] = np.multiply(tmp, Q_in[order])
                factor = np.power(Q_out, 2)-S_out
                Sigma = new_Sigma.copy()
                gamma = 1 - np.multiply(alpha, np.diag(Sigma).reshape(-1,1))
                if Likelihood == Likelihoods['GAUSSIAN']:
                    b_Basis_Phi = beta * Basis_Phi
                else:
                    b_Basis_Phi = np.multiply(Phi, beta*np.ones(1,Mt)).T*Basis
            else:
                Sigma, mu, S_in, Q_in, S_out, Q_out, factor,\
                        newLogML, gamma, b_Basis_Phi, beta = \
                        updateStats(Likelihood, Basis, Phi, \
                        targets, order, alpha, beta, mu, Basis_Phi,\
                        Basis_targets, options)
                deltaLogMarginal = newLogML - logML
            if update_iter and deltaLogMarginal<0:
                Diagnostics(options, 1, 'Warning: Marginal Likelihood decreases\
                           %g!'% deltaLogMarginal)
            logML = logML + deltaLogMarginal
            logMarginalLog[count, 0] = logML
            count = count+1
    
        '''    update Noise Parameter beta      '''
        
        if Likelihood == Likelihoods['GAUSSIAN'] and not options['fixedNoise']\
           and (selectedAction==ACTION['terminate'] or\
                it<=controls['betaUpdateStart']\
                or not it%controls['betaUpdateFrequency']):
            beta_old = beta
            y = Phi*mu
            e = targets-y
            beta = (N-np.sum(gamma))/np.power(e,2).sum()
            beta = np.min([beta, controls['betaMaxFactor']/targets.var()])
            delta_logbeta = np.log(beta/beta_old)
            if np.abs(delta_logbeta)>controls['MinDeltaLogBeta']:
                Sigma, mu, S_in, Q_in, S_out, Q_out, factor,\
                        logML, gamma, b_Basis_Phi, beta = \
                        updateStats(Likelihood, Basis, Phi, \
                        targets, order, alpha, beta, mu, Basis_Phi,\
                        Basis_targets, options)
               	count = count+1
                logMarginalLog[count,0] = logML
                if selectedAction == ACTION['terminate']:
                    selectedAction = ACTION['noise_only']
                    Diagnostics(options, 3, 'Noise update and termination defferd!')

        if selectedAction == ACTION['terminate']:
            Diagnostics(options, 2, 'Stop at iteration %d (max deltaML = %.3f)\
            '%(it, np.max(deltaLogMarginal)))
            if Likelihood == Likelihoods['GAUSSIAN']:
                Diagnostics(options, 2, '%4d>\tL=%.6f\tgamma=%.2f(M=%d)\tsigma=%.3f'\
                        %(it, logML/N, np.sum(gamma), Mt, np.sqrt(1/beta)))
            else:
                Diagnostics(options, 2, '%4d>\tL=%.6f\tgamma=%.2f(M=%d)'\
                        %(it, logML/N, np.sum(gamma), Mt))
            break

        ITER_LIMIT = it==options['max_iter']
        TIME_LIMIT = (time.time()-t_start)>=options['max_time']
        LAST_ITER = ITER_LIMIT or TIME_LIMIT
        if (options['monitor'] and not i%options['monitor']) or LAST_ITER:
            if Likelihood == Likelihoods['GAUSSIAN']:
                Diagnostics(options, 2, '%4d>\tL=%.6f\tgamma=%.2f(M=%d)\tsigma=%.3f'\
                        %(it, logML/N, np.sum(gamma), Mt, np.sqrt(1/beta)))
            else:
                Diagnositcs(options, 2, '%4d>\tL=%.6f\tgamma=%.2f(M=%d)'\
                        %(it, logML/N, np.sum(gamma), Mt))

    '''         END OF MAIN LOOP        '''
    
    if selectedAction != ACTION['terminate']:
        if ITER_LIMIT:
            Diagnostics(options, 1, 'Iteration Limit: Algorithm did not converge!')
        elif TIME_LIMIT:
            Diagnostics(options, 1, 'Iteration Limit: Algorithm did not converge!')

    if options['diagnosticLevel']>1:
        t_stop = time.time()
        total = add_count + delete_count + update_count
        if controls['BasisAlignmentTest']:
            total = total + alignDeferCount
        total = 1 if total == 0 else total
        Diagnostics(options, 2, 'Action Summary\n===============')
        Diagnostics(options, 2, 'Added\t\t%6d(%.0f%%)'%(add_count,100*add_count/total))
        Diagnostics(options, 2, 'Deleted\t\t%6d(%.0f%%)'%(delete_count,100*delete_count/total))
        Diagnostics(options, 2, 'Reesimated\t%6d(%.0f%%)'%(update_count,100*update_count/total))
        if controls['BasisAlignmentTest'] and alignDeferCount:
            Diagnostics(options, 2, '--------------')
            Diagnostics(options, 2, 'Deferred\t%6d(%.0f%%)'%(alignDeferCount,100*alignDeferCount/total))
        Diagnostics(options, 2, '==============')
        Diagnostics(options, 2, 'Total of %d likelihood updates'%count)
        Diagnostics(options, 2, 'Time to run: %.2f seconds'%(t_stop-t_start))
    Diagnostics(options, 'end')

    Param = {}
    HyperParam = {}
    Diagnostic = {}
    argorder = np.argsort(order)
    Param['relevant'] = list(used)
    Param['value'] = np.divide(mu[argorder],Scale[:,list(used)].T)
    HyperParam['alpha'] = np.divide(alpha[argorder],\
                        np.power(Scale[:,list(used)],2).T)
    HyperParam['beta'] = beta
    Diagnostic['gamma'] = gamma[argorder]
    Diagnostic['likelihood'] = logMarginalLog[1:count,0]
    Diagnostic['iterations'] = it
    Diagnostic['sparse_factor'] = S_out
    Diagnostic['quality_factor'] = Q_out

    return Param, HyperParam, Diagnostic

if __name__ == '__main__':
    a = [rd.uniform(0, 100) for i in range(20*20)]
    basis = np.matrix(a).reshape(20,20)
    w = [rd.uniform(0, 100) for i in range(20)]
    w = np.matrix(w).reshape(-1, 1)
    n = [rd.uniform(0,1) for i in range(20)]
    noise = np.matrix(n).reshape(-1, 1)
    targets = basis*w+noise
    Param, HyperParam, Diagnostic = Bayesian('gaussian', basis, targets)
