from math import ceil, floor, log, log2, sqrt, inf
import random
from numpy import pi
from numpy.lib.type_check import iscomplex
import numpy as np
from minimize import minimize



def TAFKAP_decode(samples=None, p={}):

    def fun_negLL_norm(params, getder=True):
        if getder:
            minll, minder = fun_LL_norm(params, getder)
            minder*=-1
        else:
            minll = fun_LL_norm(params, getder)
        minll*=-1        
        if getder:             
            return minll, minder
        else:
            return minll

    def fun_LL_norm(params, getder=True):
        #Computes the log likelihood of the noise parameters. Also returns 
        #the partial derivatives of each of the parameters (i.e. the 
        #gradient), which are required by minimize.m and other efficient
        #optimization algorithms.
        
        input_type ='numpy'        
        if type(params) is list:
            input_type='list'
            params = np.array(params)

        nvox = noise.shape[1]
        ntrials = noise.shape[0]
        tau = params[0:-2]
        sig = params[-2]
        rho = params[-1]

        omi, NormConst = invSNC(W[:, 0:p['nchan']], tau, sig, rho, True)

        XXt = np.matmul(noise.T, noise)
        negloglik = 0.5*(MatProdTrace(XXt, omi) + ntrials*NormConst)
        negloglik = negloglik.item()

        if iscomplex(negloglik): negloglik=inf #If we encounter a degenerate solution (indicated by complex-valued likelihoods), make sure that the likelihood goes to infinity.       

        if (np.concatenate((tau,np.expand_dims(sig,-1)),0)<0.001).any(): negloglik=inf
        if np.abs(rho) > 0.999999: negloglik=inf

        loglik = -negloglik

        if getder:
            der = np.empty_like(params)

            ss = sqrt(ntrials)
            U = np.matmul(omi, noise.T)/ss

            dom = np.matmul(omi, np.eye(nvox) - np.matmul(((1/ntrials)*XXt), omi))            

            JI = 1-np.eye(nvox)
            R = np.eye(nvox)*(1-rho)+rho
            der[0:-2] = np.matmul(2*(dom*R), tau[:,np.newaxis]).squeeze()
            der[-1] = (dom*(tau[:,np.newaxis]*tau[np.newaxis,:])*JI).sum()

            der[-2] = 2*sig*MatProdTrace(np.matmul(W[:, 0:p['nchan']].T,omi), W[:, 0:p['nchan']]) - sqrt(2*sig)*(np.matmul(U.T, W[:, 0:p['nchan']])**2).sum()

            der*=-0.5*ntrials
           
            if input_type=='list':
                der = der.tolist()

            return loglik, der
        else:
            return loglik






    def estimate_W(samples=None, C=None, do_boot=False, test_samples=None, test_C=None):
        N = C.shape[0]
        if do_boot:
            idx = np.random.randint(0,N,(N,))
        else:
            idx = np.arange(0,N)

        if p['prev_C']:
            sol = np.linalg.lstsq(np.concatenate((C[idx,p['nchan']:], np.ones((N,1))), 1), samples[idx,:])
            W_prev = sol[0].T
            W_prev = W_prev[:,0:-1]
            samples -= np.matmul(C[:, p['nchan']:], W_prev.T)
            C = C[:, 0:p['nchan']]
        else:
            W_prev = np.empty((samples.shape[1],0))

        sol = np.linalg.lstsq(C[idx,:], samples[idx,:], rcond=None)
        W_curr = sol[0].T        
        W = np.concatenate((W_curr, W_prev), 1)
        
        noise = samples[idx,:] - np.matmul(C[idx,:], W_curr.T)
        if not test_samples is None:
            test_noise = test_samples - np.matmul(test_C, W.T)
        else:
            test_noise = None

        return W, noise, test_noise

    def estimate_cov(X,lambda_var,lamb,W):
        n,pp = X.shape[:]
        W = W[:,0:p['nchan']]     
        
        vars = (X**2).mean(0)
        medVar = np.median(vars)

        t = np.tril(np.ones((pp,)*2),-1)==1        
        samp_cov = np.matmul(X.T,X)/n
        
        WWt = np.matmul(W,W.T)
        rm = np.concatenate((np.expand_dims(WWt[t], -1), np.ones((t.sum(),1))),1)
        sol = np.linalg.lstsq(rm, np.expand_dims(samp_cov[t], -1), rcond=None)
        coeff = sol[0]

        target_diag = lambda_var*medVar + (1-lambda_var)*vars
        target = coeff[0]*WWt + np.ones((pp,)*2)*coeff[1]
        target[np.eye(pp)==1]=target_diag
                
        C = (1-lamb)*samp_cov + lamb*target
        try:
            np.linalg.cholesky(C) #Cholesky decomp seems to be faster than eigendecomp, so assuming we mostly don't fail this test, it's faster this way
        except:
            eigvals, eigvecs = np.linalg.eigh(C)
            min_eigval = eigvals.min()
            print('WARNING: Non-positive definite covariance matrix detected. Lowest eigenvalue: ' + str(min_eigval.item()) + '. Finding a nearby PD matrix by thresholding eigenvalues at 1e-10.')        
            eigvals = eigvals.clamp(1e-10)
            eigvals = np.diag(eigvals)
            C = np.matmul(np.matmul(eigvecs,eigvals), eigvecs.T)

        return C

    def find_lambda(cvInd, lambda_range):        

        cv_folds = np.unique(cvInd)
        K = cv_folds.shape[0]
        assert K>1, 'Must have at least two CV folds'
        
        W_cv, est_noise_cv, val_noise_cv = [],[],[]

        def visit(intern_lamb):
            loss = 0
            for cv_iter2 in range(K):
                estC = estimate_cov(est_noise_cv[cv_iter2], intern_lamb[0], intern_lamb[1], W_cv[cv_iter2])
                vncv = val_noise_cv[cv_iter2]
                valC = np.matmul(vncv.T, vncv)/vncv.shape[0] #sample covariance of validation data
                loss += fun_norm_loss(estC, valC)

            if isinstance(loss, complex):
                loss = np.array(inf)                
            
            return loss


        # Pre-compute tuning weights and noise values to use in each cross-validation split
        for cv_iter in range(K):
            val_trials = cvInd==cv_folds[cv_iter]
            est_trials = np.logical_not(val_trials)
            est_samples = train_samples[est_trials,:]
            val_samples = train_samples[val_trials,:]
            this_W_cv, this_est_noise_cv, this_val_noise_cv = estimate_W(est_samples, Ctrain[est_trials,:,0], False, val_samples, Ctrain[val_trials,:,0])
            W_cv.append(this_W_cv)
            est_noise_cv.append(this_est_noise_cv)
            val_noise_cv.append(this_val_noise_cv)

        # Grid search
        s = [x.size for x in lambda_range]
        Ngrid = [min(max(2, ceil(sqrt(x))), x) for x in s] #Number of values to visit in each dimension (has to be at least 2, except if there is only 1 value for that dimension)        

        grid_vec = [np.linspace(0,y-1,x).astype(int) for x,y in zip(Ngrid, s)]
        grid_x, grid_y = np.meshgrid(grid_vec[0], grid_vec[1], indexing='ij')
        grid_l1, grid_l2 = np.meshgrid(lambda_range[0], lambda_range[1], indexing='ij')
        grid_l1, grid_l2 = grid_l1.flatten(), grid_l2.flatten()

        sz = s.copy()
        sz.reverse()
        
        print('--GRID SEARCH--')
        losses = np.empty((grid_x.size,1))
        for grid_iter in range(grid_x.size):
            this_lambda = np.array((lambda_range[0][grid_x.flatten()[grid_iter]], lambda_range[1][grid_y.flatten()[grid_iter]]))
            losses[grid_iter] = visit(this_lambda)
            print("{:02d}/{:02d} -- lambda_var: {:3.2f}, lambda: {:3.2f}, loss: {:g}".format(grid_iter, grid_x.size, *this_lambda, losses[grid_iter].item()))

        visited = sub2ind(sz, grid_y.flatten().tolist(), grid_x.flatten().tolist())
        best_loss, best_idx = losses.min(0).item(), losses.argmin(0).item()
        best_idx = visited[best_idx]

        best_lambda_gridsearch = (grid_l1[best_idx], grid_l2[best_idx])
        print('Best lambda setting from grid search: lambda_var = {:3.2f}, lambda = {:3.2f}, loss = {:g}'.format(*best_lambda_gridsearch, best_loss))        
                
        # Pattern search
        print('--PATTERN SEARCH--')
        step_size = int(2**floor(log2(np.diff(grid_y[0][0:2])/2))) #Round down to the nearest power of 2 (so we can keep dividing the step size in half
        while True:
            best_y, best_x = ind2sub(sz,best_idx)                        
            new_x = best_x + np.array((-1, 1, -1, 1)).astype(int)*step_size
            new_y = best_y + np.array((-1, -1, 1, 1)).astype(int)*step_size            
            del_idx = np.logical_or(np.logical_or(new_x<0, new_x >= lambda_range[0].size), np.logical_or( new_y<0,  new_y >= lambda_range[1].size))            
            new_x = new_x[np.logical_not(del_idx)]
            new_y = new_y[np.logical_not(del_idx)]
            new_idx = sub2ind(sz, new_y.tolist(), new_x.tolist())            
            not_visited = [x not in visited for x in new_idx]
            new_idx = [i for (i, v) in zip(new_idx, not_visited) if v]
            if len(new_idx)>0:
                this_losses = np.empty(len(new_idx))
                for ii in range(len(new_idx)):
                    this_lambda = np.array((grid_l1[new_idx[ii]], grid_l2[new_idx[ii]]))
                    this_losses[ii] = visit(this_lambda)
                    print("Step size: {:d}, lambda_var: {:3.2f}, lambda: {:3.2f}, loss: {:g}".format(step_size, *this_lambda, this_losses[ii].item()))
                visited.extend(new_idx)
                # visited = np.concatenate((visited, np.array(new_idx)),0)
                losses = np.concatenate((losses, this_losses[:,np.newaxis]),0)

            if (this_losses<best_loss).any():
                best_loss, best_idx = losses.min(0).item(), losses.argmin(0).item()
                best_idx = visited[best_idx]
            elif step_size>1:
                step_size = int(step_size/2)
            else:
                break

        best_lambda = np.array((grid_l1[best_idx], grid_l2[best_idx]))
        print("Best setting found: lambda_var = {:3.2f}, lambda = {:3.2f}, loss: {:g}".format(*best_lambda, best_loss))
        
        return best_lambda


    defaults = { #Default settings for parameters in 'p'    
    'Nboot': int(5e4), #Maximum number of bootstrap iterations 
    'precomp_C': 4, #How many sets of channel basis functions to use (swap between at random) - for PRINCE, this value is irrelevant
    'randseed': 1234, #The seed for the (pseudo-)random number generator, which allows the algorithm to reproduce identical results whenever it's run with the same input, despite being stochastic. 
    'prev_C': False, #Regress out contribution of previous stimulus to current-trial voxel responses?        
    'dec_type': 'TAFKAP', # 'TAFKAP' or 'PRINCE'            
    'stim_type': 'circular', #'circular' or 'categorical'. Also controls what type of data is simulated, in case no data is provided.        
    'DJS_tol': 1e-8, #If the Jensen-Shannon Divergence between the new likelihoods and the previous values is smaller than this number, we stop collecting bootstrap samples (before the maximum of Nboot is reached). If you don't want to allow this early termination, you can set this parameter to a negative value.
    'nchan': 8, #Number of "channels" i.e. orientation basis functions used to fit voxel tuning curves
    'chan_exp': 5, #Exponent to which basis functions are raised (higher = narrower)
    'precision': 'double'
    }
    
    # set default parameters
    p = setdefaults(defaults, p)   
    
    if samples is None:
        print('--SIMULATING DATA--')
        Ntraintrials = 200
        Ntesttrials = 20
        Ntrials = Ntraintrials+Ntesttrials
        nclasses = 4 # Only relevant when simulating categorical stimuli

        # simulating data
        samples, sp = makeSNCData({
            'nvox': 500, 
            'ntrials': Ntrials, 
            'taumean': 0.7, 
            'ntrials_per_run': Ntesttrials, 
            'Wstd': 0.3,
            'sigma': 0.3, 
            'randseed': p['randseed'], 
            'shuffle_oris': 1,
            'sim_stim_type': p['stim_type'],
            'nclasses': nclasses        
        })

        # set parameters with simulation parameters
        p["Ntraintrials"] = Ntraintrials
        p['stimval'] = sp['stimval']
        p['runNs'] = sp['run_idx']

    # get number of trials
    Ntrials = samples.shape[0]

    # create test and train data indices
    assert "Ntraintrials" in p, "Must specify Ntraintrials in parameters"
    p['train_trials'] = np.arange(0,Ntrials) < p["Ntraintrials"]
    p['test_trials'] = np.logical_not(p['train_trials'])

    # transform circular stimulus values to radians
    if p['stim_type'] == 'circular':
        p['stimval'] /= (pi/90)

    assert 'stimval' in p and 'runNs' in p, 'Must specify stimval and runNs'

    np.random.seed(p['randseed'])
    random.seed(p['randseed'])

    train_samples = samples[p['train_trials'],:]
    test_samples = samples[np.logical_not(p['train_trials']),:]
    Ntraintrials = train_samples.shape[0]
    Ntesttrials = test_samples.shape[0]
    Nvox = train_samples.shape[1]
    train_stimval = p['stimval'][p['train_trials']]
    if p['stim_type']=='circular': train_stimval /= (90/pi)

    
    
    del samples


    """  Pre-compute variables to speed up computation
    To speed up computation, we discretize the likelihoods into 100 equally
    spaced values (this value is hardcoded but can be changed as desired).
    This allows us to precompute the channel responses (basis function
    values) for these 100 stimulus values (orientations). 

    For categorical stimulus variables, likelihoods are discrete by
    definition, and evaluated only for the M classes that the data belong to.
    """

    if p['stim_type']=='circular':
        s_precomp = np.linspace(0, 2*pi, 101)
        s_precomp = (s_precomp[0:-1]).reshape(100,1)
        ph = np.linspace(0, 2*pi/p['nchan'], p['precomp_C']+1)
        ph = ph[0:-1]
        classes=None
    elif p['stim_type']=='categorical':
        classes = np.unique(p['stimval'])
        assert (classes==classes.astype(int)).all(), 'Class labels must be integers'
        classes = classes.astype(int)
        Nclasses = classes.size
        p['nchan']=Nclasses
        ph=np.zeros(1)
        p['precomp_C'] = 1
        s_precomp = classes.reshape(Nclasses,1)
    
    C_precomp = np.empty((s_precomp.shape[0], p['nchan'], p['precomp_C']))
    Ctrain = np.empty((Ntraintrials, p['nchan'], p['precomp_C']))
    for i in range(p['precomp_C']):
        C_precomp[:,:,i] = fun_basis(s_precomp-ph[i], p['nchan'], p['chan_exp'], classes)
        Ctrain[:,:,i] = fun_basis(train_stimval-ph[i], p['nchan'], p['chan_exp'], classes)

    Ctest = np.empty((Ntesttrials,p['nchan'],p['precomp_C']))
    if p['prev_C']:
        Ctrain_prev = np.concatenate((np.empty((1, p['nchan'], p['precomp_C'])), Ctrain[0:-1,:,:]), 0)
        train_runNs = p['runNs'][p['train_trials']]
        sr_train = train_runNs == np.concatenate((np.empty(1), train_runNs[0:-1]), 0)
        Ctrain_prev[np.logical_not(sr_train),:,:] = 0
        Ctrain = np.concatenate((Ctrain, Ctrain_prev), 1)
        test_runNs = p['runNs'][p['test_trials']]
        sr_test = test_runNs == np.concatenate((np.empty(1), test_runNs[0:-1]), 0)

        test_stimval = p['stimval'][p['test_trials']]
        if p['stim_type']=='circular': test_stimval/=(90/pi)
        for i in range(p['precomp_C']):
            Ctest[:,:,i] = fun_basis(test_stimval-ph[i], p['nchan'], p['chan_exp'], classes)
        Ctest_prev = np.concatenate((np.empty((1, p['nchan'], p['precomp_C'])), Ctest[0:-1,:,:]), 0)
        Ctest_prev[np.logical_not(sr_test),:,:] =0

    cnt = np.zeros((Ntesttrials, s_precomp.shape[0]))
    
    # Find best hyperparameter values (using inner CV-loop within the training data)
    if p['dec_type']=='TAFKAP':
        print('--PERFORMING HYPERPARAMETER SEARCH--')
        lvr = np.linspace(0,1,50)
        lr = np.linspace(0,1,50)
        lr = lr[1:]
        hypers = find_lambda(p['runNs'][p['train_trials']], (lvr, lr))
    elif p['dec_type']=='PRINCE':
        # No hyperparameter search necessary for PRINCE
        p['Nboot']=1
        hypers = None
    else:
        raise Exception('Invalid decoder type specified')


    # Bootstrap loop (run only once for PRINCE)

    for i in range(p['Nboot']):
        ## Bootstrap sample of W & covariance
        # Resample train trials with replacement and estimate W and the
        # covariance matrix on this resampled training data. For PRINCE, don't
        # bootstrap, but just estimate W and covariance matrix once for the
        # unresampled training data.

        if p['dec_type']=='TAFKAP':
            print('Bootstrap iteration: {:d}'.format(i))

            if p['precomp_C']>1:
                pc_idx = random.randint(0, p['precomp_C']-1)
            else:
                pc_idx = 0
                
            W, noise, _ = estimate_W(train_samples, Ctrain[:,:,pc_idx], True)
            cov_est = estimate_cov(noise, hypers[0], hypers[1], W)
            
            prec_mat = chol_invld(cov_est)
            if not isinstance(prec_mat, np.ndarray):
                print('WARNING: Covariance estimate wasn\'t positive definite. Trying again with another bootstrap sample.')
                continue
        
        else:
            print('--ESTIMATING PRINCE GENERATIVE MODEL PARAMETERS--')
            W, noise, _ = estimate_W(train_samples, Ctrain[:,:,0], False)            
            init_losses = np.ones(100)*inf
            while np.isinf(init_losses).all():
                inits = [np.random.rand(Nvox+2) for x in range(100)]
                init_losses = np.array([fun_negLL_norm(x,False) for x in inits])
            
            min_idx = init_losses.argmin(0)        

            sol,_,_ = minimize(inits[min_idx], fun_negLL_norm, maxnumlinesearch=1e4)            
            
            sol = np.array(sol)
            prec_mat = invSNC(W[:, 0:p['nchan']], sol[0:-2], sol[-2], sol[-1], False)
            pc_idx=0

        # Compute likelihoods on test-trials given model parameter sample

        pred = C_precomp[:,:,pc_idx] @ W[:, 0:p['nchan']].T

        if (i+1)%100==0: old_cnt = cnt.copy()

        # The following lines are a bit different (and more elegant/efficient) in Python+Pytorch than in Matlab
        res = test_samples
        if p['prev_C']:
            res -= np.matmul(Ctest_prev[:,:,pc_idx], np.expand_dims(W[:, p['nchan']:].T, 0)).squeeze()
        
        res = res[:,np.newaxis] - pred[np.newaxis,:]
        ps = -0.5*((res @ prec_mat) * res).sum(-1)
        ps = softmax(ps-ps.max(1,keepdims=True), 1)

        cnt += ps

        if (i+1)%100==0:
            mDJS = fun_DJS(old_cnt/old_cnt.sum(1,keepdims=True), cnt/cnt.sum(1,keepdims=True)).max()
            print('Max. change in likelihoods (JS-divergence) in last 100 iterations: {:g}'.format(mDJS))
            if mDJS < p['DJS_tol']: break

    liks = cnt/cnt.sum(1,keepdims=True) #(Normalized) likelihoods (= posteriors, assuming a flat prior)
    if p['stim_type'] == 'circular':        
        # if p['precision']=='double':
        #     pop_vec = liks.type(torch.complex128) @ (1j*s_precomp).exp()
        # elif p['precision']=='single':
        #     pop_vec = liks.type(torch.complex64) @ (1j*s_precomp).exp()
        pop_vec = np.matmul(liks, np.exp(1j*s_precomp))
        est = (np.angle(pop_vec)/pi*90) % 180 #Stimulus estimate (likelihood/posterior means)
        unc = np.sqrt(-2*np.log(np.abs(pop_vec)))/pi*90 #Uncertainty (defined here as circular SDs of likelihoods/posteriors)
    elif p['stim_type'] == 'categorical':
        _, est = liks.max(1)
        est = classes[est] #Convert back to original class labels
        tmp = -liks*np.log(liks)
        tmp[liks==0] = 0
        unc = tmp.sum(1) #Uncertainty (defind as the entropy of the distribution)

    return est, unc, liks, hypers
            


def fun_DKL(P,Q):
    # Computes JS-divergence between corresponding rows in P and Q
    z = P==0
    out = P*(np.log(P)-np.log(Q))
    out[z]=0
    return out.sum(1)

def fun_DJS(P,Q):
    # Computes KL-divergence from each row in Q to each corresponding row in P
    M = P/2+Q/2
    out = fun_DKL(P,M)/2 + fun_DKL(Q,M)/2
    return out

def cholesky_inverse(X):
    c = np.linalg.inv(X)
    inverse = np.dot(c.T,c)
    return inverse
    
def makeSNCData(p={}):
    

    defs = {
    'randseed': 1234,
    'Wstd': 0.3,
    'rho': 0.05,
    'sigma': 0.3,
    'nvox': 100,
    'taumean': 0.7,
    'taustd': 0.035,
    'nchan': 8,
    'nclasses': 4,
    'ntrials': 220,
    'ntrials_per_run': 20,
    'sigma_arb': 0.3,
    'sim_stim_type': 'circular'
    }

    p = setdefaults(defs,p)    
    np.random.seed(p['randseed'])
    random.seed(p['randseed'])

    nruns = p['ntrials']/p['ntrials_per_run']
    assert round(nruns)==nruns, 'Number of trials per run must divide evenly into total number of trials'
    nruns = int(nruns)    
    run_idx = np.arange(nruns).repeat(p['ntrials_per_run'])

    if p['sim_stim_type'] == 'circular':        
        base_ori = np.linspace(0, 2*pi, p['ntrials_per_run']+1)    
        base_ori = base_ori[:-1]        
        run_offsets = (np.random.rand(nruns)*(base_ori[1]-base_ori[0]))
        ori = base_ori + run_offsets[:,np.newaxis]                
        stimval = ori
        classes = None

    elif p['sim_stim_type'] == 'categorical':
        p['nchan'] = p['nclasses']
        assert (p['ntrials_per_run'] % p['nclasses'])==0, 'To simulate categorical stimulus labels, number of classes must divide neatly into the number of trials per run.'                
        stimval = np.tile(np.arange(p['nclasses']), (1, int(p['ntrials_per_run']/p['nclasses']))).repeat((nruns,1))        
        classes = np.arange(p['nclasses'])

    for j in range(nruns):
        stimval[j][:] = stimval[j][np.random.permutation(p['ntrials_per_run'])]
        
    stimval = stimval.ravel()

    # Simulate generative model parameters
    W = np.random.randn(p['nvox'],p['nchan'])*p['Wstd']
    tau_sim = np.random.randn(p['nvox'],1)*p['taustd']+p['taumean']    
    sig_sim = p['sigma']
    rho_sim = p['rho']
        
    W_shuffled = W
    for j in range(W_shuffled.shape[0]):
        W_shuffled[j][:] = W_shuffled[j][np.random.permutation(W_shuffled.shape[1])]        

    cov_sim = (1-rho_sim)*np.diag(tau_sim.squeeze()**2) + rho_sim*(tau_sim*tau_sim.T) + sig_sim**2*np.matmul(W,W.T) + p['sigma_arb']**2*np.matmul(W_shuffled,W_shuffled.T)    

    Q = np.linalg.cholesky(cov_sim)
    noise = np.matmul(Q, np.random.randn(p['nvox'], p['ntrials'])).T
    tun = np.matmul(fun_basis(stimval,classes=classes),W.T)
    rsp = tun + noise

    simpar = p
    simpar['W'] = W
    simpar['stimval'] = stimval
    simpar['tau'] = tau_sim
    simpar['rho'] = rho_sim    
    simpar['prec_mat'] = np.linalg.inv(cov_sim)
    simpar['run_idx'] = run_idx

    return rsp, simpar
    
def fun_norm_loss(c_est, c0):
    c_est_inv, c_est_ld = chol_invld(c_est, True)    

    if isinstance(c_est_inv, np.ndarray):           
        loss = (c_est_ld + (c_est_inv*c0).sum())/c0.shape[1]                        
    else:
        loss = (np.linalg.slogdet(c_est)[1] + np.linalg.solve(c_est, c0).trace())/c0.shape[1]

    return loss
 
def fun_basis(s, nchan=8.0, expo=5.0, classes=None):
    if s.ndim==1: s=s[:,np.newaxis]
    d = s.shape
    
    if d[1]>d[0]: s=s.T #make sure s is a column vector
    
    if not classes==None:
        if classes.dim()==1: classes=classes[np.newaxis,:]
        d = classes.shape        
        if d[0]>d[1]: classes=classes.T #make sure classes is a row vector
        c = (s==classes)*1.0
    else:
        TuningCentres = np.arange(0.0, 2*pi-0.001, 2*pi/nchan)
        c = np.cos(s-TuningCentres[np.newaxis,:]).clip(0)**expo

    return c

def chol_invld(X, get_ld=False):
    try:
        A = np.linalg.cholesky(X)
        Xi = cholesky_inverse(A)
        if get_ld: 
            ld = 2*np.log(np.diag(A)).sum()
            return Xi, ld
        else:
            return Xi
    except:
        if get_ld:
            return -1,-1
        else:
            return -1


def sub2ind(sz, row, col):
    n_rows = sz[0]    
    return [n_rows * c + r for r, c in zip(row, col)]
    
def ind2sub(sz, idx):
    c = int(floor(idx / sz[0]))
    r = idx % sz[0]

    return r,c


def logdet(A, try_chol=True):
    if try_chol:    
        try:
            v = 2*np.log(np.diag(np.linalg.cholesky(A))).sum()
            return v    
        except:
            v = 0 #Just to satisfy PyLance; we will fall through out of the try block anyway and compute v for real

    _,v = np.linalg.slogdet(A)
    
    return v

def invSNC(W, tau, sig, rho, getld=True):
    """
    % This code implements an efficient way of computing the inverse and log 
    % determinant of the covariance matrix used in van Bergen, Ma, Pratte & 
    % Jehee (2015), in the form of two low-rank updates of a diagonal matrix.
    %
    % Reference:
    % van Bergen, R. S., Ma, W. J., Pratte, M. S., & Jehee, J. F. M. (2015). 
    % Sensory uncertainty decoded from visual cortex predicts behavior. 
    % Nature neuroscience, 18(12), 1728-1730. https://doi.org/10.1038/nn.4150
    """

    nvox = W.shape[0]
    
    if sig==0 and rho==0:
        omi = np.diag(tau**-2)
        ld = 2*np.log(tau).sum()
    else:
        
        """"
        % Inverses of large matrices are cumbersome to compute. Our voxel
        % covariance is often a large matrix. However, we can leverage the fact
        % that this large matix has a simple structure, being the sum of a
        % diagonal matrix and two low-rank components. It turns out that under
        % these circumstances, we can compute the inverse of the full matrix by
        % first computing the inverse of the diagonal matrix (which is easy)
        % and then "updating" this inverse by the contributions of the low-rank
        % components. 
        %
        % Older versions did this in two steps: first using the
        % Sherman-Morrison identity and then updating the resulting inverse by
        % the Woodbury identity, but it turns out to be faster to do it all ina
        % single step via the Woodbury.
        %
        % More information:
        % https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf 
        % https://en.wikipedia.org/wiki/Woodbury_matrix_identity

        """         
    
        alpha = 1/(1-rho)
        Ai = alpha*np.diag(tau**-2)
        Ci = np.diag(1/np.concatenate((np.ones((1,1))*rho, np.ones((W.shape[1],1))*sig**2)).squeeze())
        U = np.concatenate((tau[:,np.newaxis],W),1)
        AiU = np.matmul(Ai,U)
        UtAiU = np.matmul(U.T,AiU)
        omi = Ai - np.matmul(np.linalg.solve(Ci+UtAiU, AiU.T).T, AiU.T)

        if getld:
            """
            % We can apply similar tricks to the log determinant. In this case,
            % this is based on the Matrix Determinant Lemma.
            % (https://en.wikipedia.org/wiki/Matrix_determinant_lemma)
            """
            ld = logdet(Ci + UtAiU) + log(rho) + W.shape[1]*2*log(sig) + nvox*log(1-rho) + 2*(np.log(tau)).sum() 
            return omi, ld
        else:
            return omi

def softmax(X, axis=1):
    X = np.exp(X)
    X /= X.sum(axis, keepdims=True)
    return X    

def MatProdTrace(A,B):
    return (A.flatten()*B.T.flatten()).sum()

def MatProdDiag(A,B):
    return (A*B.T).sum(1)

def setdefaults(defaults, x):    
    for key in defaults:
        if not key in x:
            x[key]=defaults[key]

    return x

if __name__ == "__main__":
    # TAFKAP_decode()
    TAFKAP_decode(None, {'dec_type': 'PRINCE'})