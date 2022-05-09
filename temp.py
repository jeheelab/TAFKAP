import torch

def estimate_cov(X,lambda_var,lamb,W):
    n,pp = X.shape[-2:]        
    if W.ndim==2: 
        #Insert singleton batch dimension 
        W=W.unsqueeze(0)
        X=X.unsqueeze(0)
    W = W[:,:,0:8]
    batch_size = W.shape[0] 
    
    vars = (X**2).mean(-2,True)
    medVar = vars.median(-1,True).values

    t = (torch.ones((pp,)*2).tril(-1)==1).expand(batch_size, pp, pp)
    samp_cov = torch.matmul(X.transpose(1,2),X)/n
    
    WWt = torch.matmul(W,W.transpose(1,2))        
    rm = torch.cat((WWt[t].view(batch_size, -1,1), torch.ones(batch_size, t[0,:,:].sum(), 1)),-1)
    sol = torch.linalg.lstsq(rm, samp_cov[t].view(batch_size, -1, 1))

    # WWt = torch.mm(W,W.t())
    # rm = torch.cat((WWt[t].unsqueeze(-1), torch.ones(t.sum(),1)),1)
    # sol = torch.linalg.lstsq(rm, samp_cov[t].unsqueeze(-1))

    coeff = sol[0]

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