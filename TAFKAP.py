from math import ceil, floor, log, log2, sqrt, inf
import random
from numpy import pi
from numpy.lib.type_check import iscomplex
import torch
import numpy as np
from matplotlib import pyplot as plt
import time
from minimize import minimize
from scipy.io import savemat


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
        
        input_type ='tensor'
        if type(params) is np.ndarray:
            input_type = 'numpy'
            params = torch.tensor(params)
        elif type(params) is list:
            input_type='list'
            params = torch.tensor(params)

        nvox = noise.shape[1]
        ntrials = noise.shape[0]
        tau = params[0:-2]
        sig = params[-2]
        rho = params[-1]

        omi, NormConst = invSNC(W[:, 0:p['nchan']], tau, sig, rho, True)

        XXt = torch.mm(noise.t(), noise)
        negloglik = 0.5*(MatProdTrace(XXt, omi) + ntrials*NormConst)
        negloglik = negloglik.item()

        if iscomplex(negloglik): negloglik=inf #If we encounter a degenerate solution (indicated by complex-valued likelihoods), make sure that the likelihood goes to infinity.       

        if (torch.cat((tau,sig.unsqueeze(-1)),0)<0.001).any(): negloglik=inf
        if rho.abs() > 0.999999: negloglik=inf

        loglik = -negloglik

        if getder:
            der = torch.empty_like(params)

            ss = sqrt(ntrials)
            U = torch.mm(omi, noise.t())/ss

            dom = torch.mm(omi, torch.eye(nvox) - torch.mm(((1/ntrials)*XXt), omi))            

            JI = 1-torch.eye(nvox)
            R = torch.eye(nvox)*(1-rho)+rho
            der[0:-2] = torch.mm(2*(dom*R), tau.unsqueeze(-1)).squeeze()
            der[-1] = (dom*(tau.unsqueeze(-1)*tau.unsqueeze(0))*JI).sum()

            der[-2] = 2*sig*MatProdTrace(torch.mm(W[:, 0:p['nchan']].t(),omi), W[:, 0:p['nchan']]) - sqrt(2*sig)*(torch.mm(U.t(), W[:, 0:p['nchan']])**2).sum()

            der*=-0.5*ntrials

            if input_type=='numpy':
                der = der.numpy()
            elif input_type=='list':
                der = der.tolist()

            return loglik, der
        else:
            return loglik






    def estimate_W(samples=None, C=None, do_boot=50, test_samples=None, test_C=None):
        N = C.shape[0]        
        if do_boot>0:
            #Generate multiple bootstraps at once in a batch
            idx = torch.randint(N, (N,do_boot))
        else:
            idx = torch.arange(0,N).unsqueeze(-1)

        C_boot = C[idx,:].transpose(0,1)
        samples_boot = samples[idx,:].transpose(0,1)


        if p['prev_C']:
            sol = torch.linalg.lstsq(torch.cat((C_boot[:,:,p['nchan']:], torch.ones(do_boot, N,1)), 2), samples_boot)
            W_prev = sol[0].transpose(1,2)
            W_prev = W_prev[:,:,:-1]
            samples -= C_boot[:,:,p['nchan']:] @ W_prev.transpose(1,2)
            C_boot = C_boot[:, 0:p['nchan']]

            # sol = torch.linalg.lstsq(torch.cat((C[idx,p['nchan']:], torch.ones(N,1)), 1), samples[idx,:])
            # W_prev = sol[0].t()
            # W_prev = W_prev[:,0:-1]
            # samples -= torch.mm(C[:, p['nchan']:], W_prev.t())
            # C = C[:, 0:p['nchan']]
        else:
            W_prev = torch.empty(0)

        sol = torch.linalg.lstsq(C_boot, samples_boot)
        W_curr = sol[0].transpose(1,2)
        W = torch.cat((W_curr, W_prev), -1)

        noise = samples_boot - torch.matmul(C_boot, W_curr.transpose(1,2))        
        if not test_samples==None:
            test_noise = test_samples.unsqueeze(0) - torch.matmul(test_C.unsqueeze(0), W.transpose(1,2))
            test_noise = test_noise.squeeze()
        else:
            test_noise = None
        
        W, noise = W.squeeze(), noise.squeeze()
        return W, noise, test_noise

    def estimate_cov(X,lambda_var,lamb,W):
        n,pp = X.shape[-2:]        
        if W.ndim==2: 
            #Insert singleton batch dimension 
            W=W.unsqueeze(0)
            X=X.unsqueeze(0)
        W = W[:,:,0:p['nchan']]    
        batch_size = W.shape[0] 
        
        vars = (X**2).mean(-2,True)
        medVar = vars.median(-1,True).values

        t = (torch.ones((pp,)*2).tril(-1)==1).expand(batch_size, pp, pp)
        samp_cov = torch.matmul(X.transpose(1,2),X)/n
        
        WWt = torch.matmul(W,W.transpose(1,2))        
        rm = torch.cat((WWt[t].view(batch_size, -1,1), torch.ones(batch_size, t[0,:,:].sum(), 1)),-1)
        sol = torch.linalg.lstsq(rm, samp_cov[t].view(batch_size, -1, 1))
        coeff = sol[0]

        # coeff = torch.matmul(samp_cov[t].view(batch_size, -1, 1).transpose(-2,-1), rm.pinv().transpose(-2,1))

        target_diag = lambda_var*medVar + (1-lambda_var)*vars
        target = coeff[:,(0,),:]*WWt + torch.ones((batch_size, *(pp,)*2))*coeff[:, (1,), :]
        # target[torch.eye(pp)==1]=target_diag
        target[torch.eye(pp).expand((batch_size, *(pp,)*2))==1]=target_diag.flatten()
                
        C = (1-lamb)*samp_cov + lamb*target
        for i in range(C.shape[0]):
            #Not sure how best to batch this as the PD-test should only be done on each matrix separately
            try:
                torch.linalg.cholesky(C[i,:,:]) #Cholesky decomp seems to be faster than eigendecomp, so assuming we mostly don't fail this test, it's faster this way
            except:
                eigvals, eigvecs = torch.linalg.eigh(C[i,:,:])
                min_eigval = eigvals.min()
                print('WARNING: Non-positive definite covariance matrix detected. Lowest eigenvalue: ' + str(min_eigval.item()) + '. Finding a nearby PD matrix by thresholding eigenvalues at 1e-10.')        
                eigvals = eigvals.clamp(1e-10)
                eigvals = torch.diag(eigvals)
                C[i,:,:] = torch.mm(torch.mm(eigvecs,eigvals), eigvecs.t())

        return C.squeeze()

    def find_lambda(cvInd, lambda_range):        

        cv_folds = cvInd.unique()
        K = cv_folds.shape[0]
        assert K>1, 'Must have at least two CV folds'
        
        W_cv, est_noise_cv, val_noise_cv = [],[],[]

        def visit(intern_lamb):
            loss = 0
            for cv_iter2 in range(K):
                estC = estimate_cov(est_noise_cv[cv_iter2], intern_lamb[0], intern_lamb[1], W_cv[cv_iter2])
                vncv = val_noise_cv[cv_iter2]
                valC = torch.mm(vncv.t(), vncv)/vncv.shape[0] #sample covariance of validation data
                loss += fun_norm_loss(estC, valC)

            if loss.is_complex():
                loss = torch.Tensor(inf)                
            
            return loss


        # Pre-compute tuning weights and noise values to use in each cross-validation split
        for cv_iter in range(K):
            val_trials = cvInd==cv_folds[cv_iter]
            est_trials = val_trials.logical_not()
            est_samples = train_samples[est_trials,:]
            val_samples = train_samples[val_trials,:]
            this_W_cv, this_est_noise_cv, this_val_noise_cv = estimate_W(est_samples, Ctrain[est_trials,:,0], False, val_samples, Ctrain[val_trials,:,0])
            W_cv.append(this_W_cv)
            est_noise_cv.append(this_est_noise_cv)
            val_noise_cv.append(this_val_noise_cv)

        # Grid search
        s = [x.numel() for x in lambda_range]
        Ngrid = [min(max(2, ceil(sqrt(x))), x) for x in s] #Number of values to visit in each dimension (has to be at least 2, except if there is only 1 value for that dimension)        

        grid_vec = [torch.linspace(0,y-1,x).int() for x,y in zip(Ngrid, s)]
        grid_x, grid_y = torch.meshgrid(grid_vec[0], grid_vec[1])
        grid_l1, grid_l2 = torch.meshgrid(lambda_range[0], lambda_range[1])
        grid_l1, grid_l2 = grid_l1.flatten(), grid_l2.flatten()

        sz = s.copy()
        sz.reverse()
        
        print('--GRID SEARCH--')
        losses = torch.empty(grid_x.numel(),1)
        for grid_iter in range(grid_x.numel()):
            this_lambda = torch.Tensor((lambda_range[0][grid_x.flatten()[grid_iter]], lambda_range[1][grid_y.flatten()[grid_iter]]))
            losses[grid_iter] = visit(this_lambda)
            print("{:02d}/{:02d} -- lambda_var: {:3.2f}, lambda: {:3.2f}, loss: {:g}".format(grid_iter, grid_x.numel(), *this_lambda, losses[grid_iter].item()))

        visited = sub2ind(sz, grid_y.flatten().tolist(), grid_x.flatten().tolist())
        best_loss, best_idx = losses.min(0)
        best_idx = visited[best_idx]

        best_lambda_gridsearch = (grid_l1[best_idx], grid_l2[best_idx])
        print('Best lambda setting from grid search: lambda_var = {:3.2f}, lambda = {:3.2f}, loss = {:g}'.format(*best_lambda_gridsearch, best_loss.item()))        
                
        # Pattern search
        print('--PATTERN SEARCH--')
        step_size = int(2**floor(log2(torch.diff(grid_y[0][0:2])/2))) #Round down to the nearest power of 2 (so we can keep dividing the step size in half
        while True:
            best_y, best_x = ind2sub(sz,best_idx)                        
            new_x = best_x + torch.Tensor((-1, 1, -1, 1)).int()*step_size
            new_y = best_y + torch.Tensor((-1, -1, 1, 1)).int()*step_size            
            del_idx = torch.logical_or(torch.logical_or(new_x<0, new_x >= lambda_range[0].numel()), torch.logical_or( new_y<0,  new_y >= lambda_range[1].numel()))            
            new_x = new_x[del_idx.logical_not()]
            new_y = new_y[del_idx.logical_not()]
            new_idx = sub2ind(sz, new_y.tolist(), new_x.tolist())            
            not_visited = [x not in visited for x in new_idx]
            new_idx = [i for (i, v) in zip(new_idx, not_visited) if v]
            if len(new_idx)>0:
                this_losses = torch.empty(len(new_idx))
                for ii in range(len(new_idx)):
                    this_lambda = torch.Tensor((grid_l1[new_idx[ii]], grid_l2[new_idx[ii]]))
                    this_losses[ii] = visit(this_lambda)
                    print("Step size: {:d}, lambda_var: {:3.2f}, lambda: {:3.2f}, loss: {:g}".format(step_size, *this_lambda, this_losses[ii].item()))
                visited.extend(new_idx)
                # visited = torch.cat((visited, torch.tensor(new_idx)),0)
                losses = torch.cat((losses, this_losses.unsqueeze(-1)),0)

            if (this_losses<best_loss).any():
                best_loss, best_idx = losses.min(0)
                best_idx = visited[best_idx]
            elif step_size>1:
                step_size = int(step_size/2)
            else:
                break

        best_lambda = torch.Tensor((grid_l1[best_idx], grid_l2[best_idx]))
        print("Best setting found: lambda_var = {:3.2f}, lambda = {:3.2f}, loss: {:g}".format(*best_lambda, best_loss.item()))
        
        return best_lambda


        
    


    defaults = { #Default settings for parameters in 'p'    
    'Nboot': int(5e4), #Maximum number of bootstrap iterations 
    'precomp_C': 4, #How many sets of channel basis functions to use (swap between at random) - for PRINCE, this value is irrelevant
    'randseed': 1234, #The seed for the (pseudo-)random number generator, which allows the algorithm to reproduce identical results whenever it's run with the same input, despite being stochastic. 
    'prev_C': False, #Regress out contribution of previous stimulus to current-trial voxel responses?        
    'dec_type': 'TAFKAP', # 'TAFKAP' or 'PRINCE'            
    'stim_type': 'circular', #'circular' or 'categorical'. Also controls what type of data is simulated, in case no data is provided.        
    'DJS_tol': 1e-8, #If the Jensen-Shannon Divergence between the new likelihoods and the previous values is smaller than this number, we stop collecting bootstrap samples (before the maximum of Nboot is reached). If you don't want to allow this early termination, you can set this parameter to a negative value.
    # When boot_batch_size > 1, the DJS-tolerance gets multiplied by the batch size, since changes will tend to be proportionally larger with larger batches
    'nchan': 8, #Number of "channels" i.e. orientation basis functions used to fit voxel tuning curves
    'chan_exp': 5, #Exponent to which basis functions are raised (higher = narrower)
    'use_gpu': True, #Make use of CUDA-enabled GPU to accelerate computation?
    'boot_batch_size': 20, #Batch size for bootstraps. Instead of doing 1 bootstrap at a time, we'll do this many at once. This can further speed up computation when using GPU.
    'precision': 'single' #Tensor precision ('double' or 'single')
    }
    
    p = setdefaults(defaults, p)   
    
    if p['use_gpu']:
        if p['precision']=='double':
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        elif p['precision']=='single':
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        if p['precision']=='double':
            torch.set_default_dtype(torch.float64)
        elif p['precision']=='single':
            torch.set_default_dtype(torch.float32)
        
    
    if samples == None:
        print('--SIMULATING DATA--')
        Ntraintrials = 200
        Ntesttrials = 20
        Ntrials = Ntraintrials+Ntesttrials
        nclasses = 4; #Only relevant when simulating categorical stimuli
        
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
        
    
    p['train_trials'] = torch.arange(Ntrials) < Ntraintrials
    p['test_trials'] = torch.logical_not(p['train_trials'])

    p['stimval'] = sp['stimval']
    if p['stim_type']=='circular': p['stimval'] /= (pi/90)
    p['runNs'] = sp['run_idx'];    

    assert 'stimval' in p and 'train_trials' in p and 'test_trials' in p and 'runNs' in p, 'Must specify stimval, train_trials, test_trials and runNs'

    torch.manual_seed(p['randseed'])    
    np.random.seed(p['randseed'])
    random.seed(p['randseed'])
    # torch.use_deterministic_algorithms(True)

    train_samples = samples[p['train_trials'],:]
    test_samples = samples[torch.logical_not(p['train_trials']),:]
    Ntraintrials = train_samples.shape[0]
    Ntesttrials = test_samples.shape[0]
    Nvox = train_samples.shape[1]
    train_stimval = p['stimval'][p['train_trials']]
    if p['stim_type']=='circular': train_stimval /= (90/pi)

    DJS_interval = ceil(100/p['boot_batch_size'])*p['boot_batch_size']
    
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
        s_precomp = torch.linspace(0, 2*pi, 101)
        s_precomp = (s_precomp[0:-1]).view(100,1)
        ph = torch.linspace(0, 2*pi/p['nchan'], p['precomp_C']+1)
        ph = ph[0:-1]
        classes=None
    elif p['stim_type']=='categorical':
        classes = torch.unique(p['stimval'])
        assert (classes==classes.int()).all(), 'Class labels must be integers'
        classes = classes.int()
        Nclasses = classes.numel()
        p['nchan']=Nclasses
        ph=torch.zeros(1)
        p['precomp_C'] = 1
        s_precomp = classes.view(Nclasses,1)
    
    C_precomp = torch.empty(s_precomp.shape[0], p['nchan'], p['precomp_C'])
    Ctrain = torch.empty(Ntraintrials, p['nchan'], p['precomp_C'])
    for i in range(p['precomp_C']):
        C_precomp[:,:,i] = fun_basis(s_precomp-ph[i], p['nchan'], p['chan_exp'], classes)
        Ctrain[:,:,i] = fun_basis(train_stimval-ph[i], p['nchan'], p['chan_exp'], classes)

    Ctest = torch.empty(Ntesttrials,p['nchan'],p['precomp_C'])
    if p['prev_C']:
        Ctrain_prev = torch.cat((torch.empty(1, p['nchan'], p['precomp_C']), Ctrain[0:-1,:,:]), 0)
        train_runNs = p['runNs'][p['train_trials']]
        sr_train = train_runNs == torch.cat((torch.empty(1), train_runNs[0:-1]), 0)
        Ctrain_prev[sr_train.logical_not(),:,:] = 0
        Ctrain = torch.cat((Ctrain, Ctrain_prev), 1)
        test_runNs = p['runNs'][p['test_trials']]
        sr_test = test_runNs == torch.cat((torch.empty(1), test_runNs[0:-1]), 0)

        test_stimval = p['stimval'][p['test_trials']]
        if p['stim_type']=='circular': test_stimval/=(90/pi)
        for i in range(p['precomp_C']):
            Ctest[:,:,i] = fun_basis(test_stimval-ph[i], p['nchan'], p['chan_exp'], classes)
        Ctest_prev = torch.cat((torch.empty(1, p['nchan'], p['precomp_C']), Ctest[0:-1,:,:]), 0)
        Ctest_prev[sr_test.logical_not(),:,:] =0

    cnt = torch.zeros(Ntesttrials, s_precomp.shape[0])
    
    # Find best hyperparameter values (using inner CV-loop within the training data)
    if p['dec_type']=='TAFKAP':
        print('--PERFORMING HYPERPARAMETER SEARCH--')
        lvr = torch.linspace(0,1,50)
        lr = torch.linspace(0,1,50)
        lr = lr[1:]
        hypers = find_lambda(p['runNs'][p['train_trials']], (lvr, lr))
    elif p['dec_type']=='PRINCE':
        # No hyperparameter search necessary for PRINCE
        p['Nboot']=1
        hypers = None
    else:
        raise Exception('Invalid decoder type specified')


    # Bootstrap loop (run only once for PRINCE)

    for i in range(0, p['Nboot'], p['boot_batch_size']):
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
                
            W, noise, _ = estimate_W(train_samples, Ctrain[:,:,pc_idx], do_boot=p['boot_batch_size'])
            cov_est = estimate_cov(noise, hypers[0], hypers[1], W)
            
            prec_mat = chol_invld(cov_est)
            if not torch.is_tensor(prec_mat):
                print('WARNING: Covariance estimate wasn\'t positive definite. Trying again with another bootstrap sample.')
                continue
        
        else:
            print('--ESTIMATING PRINCE GENERATIVE MODEL PARAMETERS--')
            W, noise, _ = estimate_W(train_samples, Ctrain[:,:,0], False)            
            init_losses = torch.ones(100)*inf
            while init_losses.isinf().all():
                inits = [torch.rand(Nvox+2) for x in range(100)]
                # inits[-1] = torch.cat((torch.ones(Nvox)*0.7, torch.ones(1)*0.3, torch.ones(1)*0.05),0)
                init_losses = torch.tensor([fun_negLL_norm(x,False) for x in inits])
            
            _,min_idx = init_losses.min(0)        

            sol,_,_ = minimize(inits[min_idx].numpy(), fun_negLL_norm, maxnumlinesearch=1e4)            
            # savemat('tmp.mat', {'W':W.numpy(), 'noise':noise.numpy(), 'init':inits[min_idx].numpy(), 'sol':sol})
            sol = torch.tensor(sol)
            prec_mat = invSNC(W[:, 0:p['nchan']], sol[0:-2], sol[-2], sol[-1], False)
            pc_idx=0

        # Compute likelihoods on test-trials given model parameter sample
        if W.ndim==2:
            W = W.unsqueeze(0)
            prec_mat = prec_mat.unsqueeze(0)        
        res = test_samples.unsqueeze(0) #Dimensions will be [bootstrap_batch x test_batch x ...]
        pred = C_precomp[:,:,pc_idx] @ W[:, :, 0:p['nchan']].transpose(-2,-1)

        if all((p['boot_batch_size']==1, (i+1)%DJS_interval==0)) or all((i>0, p['boot_batch_size']>1, (i%DJS_interval)==0)): old_cnt = cnt.clone()
        
        # The following lines are a bit different (and more elegant/efficient) in Python+Pytorch than in Matlab
        
        if p['prev_C']:
            res -= torch.matmul(Ctest_prev[:,:,pc_idx], W[:, :, p['nchan']:].t().unsqueeze(0)).squeeze()
        
        res = res.unsqueeze(2) - pred.unsqueeze(1)
        ps = -0.5*((res @ prec_mat.unsqueeze(1) ) * res).sum(-1)
        ps = (ps-ps.amax(-1,True)).softmax(-1)

        cnt += ps.sum(0)

        if all((p['boot_batch_size']==1, (i+1)%DJS_interval==0)) or all((i>0, p['boot_batch_size']>1, (i%DJS_interval)==0)):
            mDJS = fun_DJS(old_cnt/old_cnt.sum(-1,True), cnt/cnt.sum(-1,True)).amax()
            print('Max. change in likelihoods (JS-divergence) since previous batch: {:g}'.format(mDJS))
            if mDJS < p['DJS_tol']*p['boot_batch_size']: break

    liks = cnt/cnt.sum(1,True) #(Normalized) likelihoods (= posteriors, assuming a flat prior)
    if p['stim_type'] == 'circular':        
        if p['precision']=='double':
            pop_vec = liks.type(torch.complex128) @ (1j*s_precomp).exp()
        elif p['precision']=='single':
            pop_vec = liks.type(torch.complex64) @ (1j*s_precomp).exp()
        est = (pop_vec.angle()/pi*90) % 180 #Stimulus estimate (likelihood/posterior means)
        unc = (-2*pop_vec.abs().log()).sqrt()/pi*90 #Uncertainty (defined here as circular SDs of likelihoods/posteriors)
    elif p['stim_type'] == 'categorical':
        _, est = liks.max(1)
        est = classes[est] #Convert back to original class labels
        tmp = -liks*liks.log() 
        tmp[liks==0] = 0
        unc = tmp.sum(1) #Uncertainty (defind as the entropy of the distribution)

    return est, unc, liks, hypers
            


def fun_DKL(P,Q):
    # Computes KL-divergence from each row in Q to each corresponding row in P    
    # For >2-dimensional tensors, 'rows' are the final dimension (leading dims are interpreted as batch dimensions)
    z = P==0
    out = P*(P.log()-Q.log())
    out[z]=0
    return out.sum(-1)

def fun_DJS(P,Q):
    # Computes JS-divergence between corresponding rows in P and Q
    # For >2-dimensional tensors, 'rows' are the final dimension (leading dims are interpreted as batch dimensions)
    M = P/2+Q/2
    out = fun_DKL(P,M)/2 + fun_DKL(Q,M)/2
    return out
    
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
    torch.manual_seed(p['randseed'])
    np.random.seed(p['randseed'])
    random.seed(p['randseed'])

    nruns = p['ntrials']/p['ntrials_per_run']
    assert round(nruns)==nruns, 'Number of trials per run must divide evenly into total number of trials'
    nruns = int(nruns)    
    run_idx = torch.arange(nruns).repeat_interleave(p['ntrials_per_run'])

    if p['sim_stim_type'] == 'circular':        
        base_ori = torch.linspace(0, 2*pi, p['ntrials_per_run']+1)    
        base_ori = base_ori[0:-1]        
        run_offsets = (torch.rand(nruns)*(base_ori[1]-base_ori[0])).unsqueeze(1)
        ori = base_ori + run_offsets                
        stimval = ori
        classes = None

    elif p['sim_stim_type'] == 'categorical':
        p['nchan'] = p['nclasses']
        assert (p['ntrials_per_run'] % p['nclasses'])==0, 'To simulate categorical stimulus labels, number of classes must divide neatly into the number of trials per run.'                
        stimval = torch.tile(torch.arange(p['nclasses']), (1, int(p['ntrials_per_run']/p['nclasses']))).repeat((nruns,1))        
        classes = torch.arange(p['nclasses'])

    for j in range(nruns):
        stimval[j][:] = stimval[j][torch.randperm(p['ntrials_per_run'])]
        
    stimval = stimval.ravel()

    # Simulate generative model parameters
    W = torch.randn((p['nvox'],p['nchan']))*p['Wstd']
    tau_sim = torch.randn((p['nvox'],1))*p['taustd']+p['taumean']    
    sig_sim = p['sigma']
    rho_sim = p['rho']
        
    W_shuffled = W
    for j in range(W_shuffled.shape[0]):
        W_shuffled[j][:] = W_shuffled[j][torch.randperm(W_shuffled.shape[1])]        

    cov_sim = (1-rho_sim)*torch.diag(tau_sim.squeeze()**2) + rho_sim*(tau_sim*tau_sim.t()) + sig_sim**2*torch.mm(W,W.t()) + p['sigma_arb']**2*torch.mm(W_shuffled,W_shuffled.t())    

    Q = torch.linalg.cholesky(cov_sim)
    noise = torch.matmul(Q, torch.randn((p['nvox'], p['ntrials']))).t()
    tun = torch.matmul(fun_basis(stimval,classes=classes),W.t())
    rsp = tun + noise

    simpar = p
    simpar['W'] = W
    simpar['stimval'] = stimval
    simpar['tau'] = tau_sim
    simpar['rho'] = rho_sim    
    simpar['prec_mat'] = torch.cholesky_inverse(cov_sim)
    simpar['run_idx'] = run_idx

    return rsp, simpar
    
def fun_norm_loss(c_est, c0):
    c_est_inv, c_est_ld = chol_invld(c_est, True)    

    if torch.is_tensor(c_est_inv):           
        loss = (c_est_ld + (c_est_inv*c0).sum())/c0.shape[1]                        
    else:
        loss = (torch.logdet(c_est) + torch.linalg.solve(c_est, c0).trace())/c0.shape[1]

    return loss
 
def fun_basis(s, nchan=8.0, expo=5.0, classes=None):
    if s.dim()==1: s=s.unsqueeze(-1)
    d = s.shape
    
    if d[1]>d[0]: s=s.t() #make sure s is a column vector
    
    if not classes==None:
        if classes.dim()==1: classes=classes.unsqueeze(0)
        d = classes.shape        
        if d[0]>d[1]: classes=classes.t() #make sure classes is a row vector
        c = (s==classes)*1.0
    else:
        TuningCentres = torch.arange(0.0, 2*pi-0.001, 2*pi/nchan).unsqueeze(0)        
        c = torch.cos(s-TuningCentres).clamp(0)**expo

    return c

def chol_invld(X, get_ld=False):
    try:
        A = torch.linalg.cholesky(X)
        Xi = torch.cholesky_inverse(A)
        if get_ld: 
            ld = 2*torch.diagonal(A,0,-2,-1).log().sum(-1)
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
            v = 2*torch.diag(torch.linalg.cholesky(A)).log().sum()
            return v    
        except:
            v = 0 #Just to satisfy PyLance; we will fall through out of the try block anyway and compute v for real

    v = torch.logdet(A)
    
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
        omi = torch.diag(tau**-2)
        ld = 2*tau.log().sum()
    else:
        """         
        % Inverses of large matrices are cumbersome to compute. Our voxel
        % covariance is often a large matrix. However, we can leverage the fact
        % that this large matix has a simple structure, being the sum of a
        % diagonal matrix and two low-rank components. It turns out that under
        % these circumstances, we can compute the inverse of the full matrix by
        % first computing the inverse of the diagonal matrix (which is easy)
        % and then "updating" this inverse by the contributions of the low-rank
        % components.
        %
        % Our two low-rank components are (1-rho)*tau*tau' and sigma²*W*W'. The 
        % former is rank-1, and so the inverse of the diagonal plus this rank-1 
        % component can be computed using the Sherman-Morrison formula. The 
        % latter has rank equal to the number of columns of W. To bring this 
        % second update into the inverse, we apply the Woodbury identity (of 
        % which Sherman-Morrison is a special case).  
        %
        % More information:
        % https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf 
        % https://en.wikipedia.org/wiki/Woodbury_matrix_identity
        % https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula 
        """

        alpha = 1/(1-rho)
        Fi = alpha*torch.diag(tau**-2)
        ti = (1/tau).unsqueeze(-1)
        Di = Fi - (rho*alpha**2*torch.mm(ti,ti.t()))/(1+rho*nvox*alpha) #Sherman-Morrison
        DiW = torch.mm(Di,W)
        WtDiW = torch.mm(W.t(),DiW)
        omi = Di - torch.mm(torch.linalg.solve((sig**-2*torch.eye(W.shape[1])+WtDiW), DiW.t()).t(), DiW.t()) #Woodbury

        if getld:
            """
            % We can apply similar tricks to the log determinant. In this case,
            % this is based on the Matrix Determinant Lemma.
            % (https://en.wikipedia.org/wiki/Matrix_determinant_lemma)
            """
            ld = logdet(torch.eye(W.shape[1]) + sig**2*WtDiW) + log(1+rho*nvox*alpha) + nvox*log(1-rho) + 2*tau.log().sum()
            return omi, ld
        else:
            return omi

def MatProdTrace(A,B):
    return (A.flatten()*B.t().flatten()).sum()

def MatProdDiag(A,B):
    return (A*B.t()).sum(1)

def setdefaults(defaults, x):    
    for key in defaults:
        if not key in x:
            x[key]=defaults[key]

    return x

if __name__ == "__main__":
    TAFKAP_decode()