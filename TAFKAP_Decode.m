function [est, unc, liks, hypers] = TAFKAP_Decode(samples, p)

%     This code runs the PRINCE or TAFKAP decoding algorithm. The original
%     PRINCE algorithm was first described here: 
% 
%     https://www.nature.com/articles/nn.4150
% 
%     The new TAFKAP algorithm is described here:
% 
%     https://www.biorxiv.org/content/10.1101/2021.03.04.433946v1
% 
%     The main input to the code is 'samples'. If you are using the code to
%     decode fMRI data, the rows in this matrix will likely correspond to
%     trials, while the columns will correspond to voxels. The variable 
%     names used in the code reflect this usage. However, note that the different
%     columns could equally be different EEG/MEG channels, or any other set
%     of variables that you measured. Similarly, rows do not have to be
%     trials, but could be any set of measurements that you took of your
%     voxels or other variables. 
% 
%     The second input, 'p', is a struct that allows you to customize        
%     certain settings that influence how the algorithm runs. These
%     settings are explained in more detail below, when their default
%     values are defined. In this struct, you must also supply some labels
%     for your data, namely a list of stimulus values ('stimval'), a list
%     of "run numbers" ('runNs') and two lists of binary indicator
%     variables ('train_trials' and 'test_trials'), that tell the code 
%     which trials to use for training, and which to use for testing. The
%     stimulus values in 'stimval' must be circular and in the range of 
%     [0, 180]. For instance, it could be the orientation of a visual
%     stimulus, measured in degrees. However, it could also be (for
%     instance) a color value or direction of motion - as long as these
%     values are rescaled to [0, 180] (just take care, in that case, to
%     transform the decoder outputs back to their original scale, e.g. [0, 360]
%     or [0, 2*pi]. Non-circular or discrete values aren't implemented
%     here, but the code could be easily adapted for these cases, by 
%     altering the basis functions (defined in fun_basis.m) that are used 
%     to fit tuning functions to voxels (or other response variables). The
%     indices in 'runNs' can correspond to the indices of the fMRI runs
%     from which each trial was taken. More broadly, they serve as indices
%     to set up an inner cross-validation loop within the training data, to
%     find the best hyperparameters for the TAFKAP algorithm (for PRINCE, 
%     these indices do not need to be specified. If your data are not
%     divided into fMRI runs, you should choose another way to divide your
%     data into independent partitions. 
% 
%     To see how the code works, you can run it without any input, in which
%     case some data will be simulated for you. The function returns four
%     outputs: 
%     -'est': an array of stimulus estimates (one for each trial)
%     -'unc': an array of uncertainty values (one for each trial)
%     -'liks': a [trial x stimulus_value] matrix of normalized
%     likelihoods/posteriors
%     -'hypers': the best setting for the hyperparameters (lambda_var and
%     lambda) that was found on the training data (not applicable to PRINCE) 
% 
%     If your use of this code leads to some form of publication, please
%     cite one of the following papers for attribution:
% 
%     For PRINCE:
%     van Bergen, R. S., Ma, W. J., Pratte, M. S., & Jehee, J. F. M. (2015). 
%     Sensory uncertainty decoded from visual cortex predicts behavior. 
%     Nature neuroscience, 18(12), 1728-1730. https://doi.org/10.1038/nn.4150
% 
%     For TAFKAP:
%     van Bergen, R.S., & Jehee, J. F. M. (2021). TAFKAP: An improved 
%     method for probabilistic decoding of cortical activity. bioRxiv, 
%     https://doi.org/10.1101/2021.03.04.433946
%
%
%     ----
%
%     Copyright (C) 2021  Ruben van Bergen & Janneke Jehee
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <https://www.gnu.org/licenses/>.

%% Initialize parameters

if nargin < 2, p = []; end
if nargin < 1, samples = []; end

defaults = { %Default settings for parameters in 'p'    
    'Nboot', 5e4; %Maximum number of bootstrap iterations 
    'precomp_C', 4; %How many sets of channel basis functions to use (swap between at random) - for PRINCE, this value is irrelevant
    'randseed', 1234; %The seed for the (pseudo-)random number generator, which allows the algorithm to reproduce identical results whenever it's run with the same input, despite being stochastic. 
    'prev_C', false; %Regress out contribution of previous stimulus to current-trial voxel responses?        
    'dec_type', 'TAFKAP'; % 'TAFKAP' or 'PRINCE'        
    'DJS_tol', 1e-8; %If the Jensen-Shannon Divergence between the new likelihoods and the previous values is smaller than this number, we stop collecting bootstrap samples (before the maximum of Nboot is reached). If you don't want to allow this early termination, you can set this parameter to a negative value.
    'nchan', 8; %Number of "channels" i.e. orientation basis functions used to fit voxel tuning curves
    'chan_exp', 5; %Exponent to which basis functions are raised (higher = narrower)
    };
p = setdefaults(defaults, p);


if isempty(samples)
    % If no data was supplied, let's simulate some data to decode
    fprintf('\n--SIMULATING DATA--\n')
    Ntraintrials = 200;
    Ntesttrials = 20;
    Ntrials = Ntraintrials+Ntesttrials;
    
    [samples, sp] = makeSNCData(struct('nvox', 500, 'ntrials', Ntrials, 'taumean', 0.7, 'ntrials_per_run', Ntesttrials, ...
        'Wstd', 0.3, 'sigma', 0.3, 'randseed', p.randseed));    
    
    
    p.train_trials = (1:Ntrials)'<= Ntraintrials;
    p.test_trials = ~p.train_trials;
    p.stimval = sp.ori/pi*90;    
    p.runNs = sp.run_idx;    
end

assert(isfield(p, 'stimval') && isfield(p, 'train_trials') && isfield(p, 'test_trials') && isfield(p, 'runNs'), 'Must specify stimval, train_trials, test_trials and runNs');

try
    rng(p.randseed, 'twister');    
catch
    rs = RandStream('mt19937ar', 'Seed', p.randseed); 
    RandStream.setDefaultStream(rs);    
end

train_samples = samples(p.train_trials,:);
test_samples = samples(p.test_trials,:);
Ntraintrials = size(train_samples,1);
Ntesttrials = size(test_samples,1);
Nvox = size(train_samples,2);
train_ori = p.stimval(p.train_trials)/90*pi;

clear samples



%% Pre-compute variables to speed up computation
% To speed up computation, we discretize the likelihoods into 100 equally
% spaced values (this value is hardcoded but can be changed as desired).
% This allows us to precompute the channel responses (basis function
% values) for these 100 stimulus values (orientations). 

s_precomp = linspace(0, 2*pi, 101)'; s_precomp(end) = [];
C_precomp = nan(length(s_precomp), p.nchan, p.precomp_C);
Ctrain = nan(Ntraintrials, p.nchan, p.precomp_C);    
ph = linspace(0, 2*pi/p.nchan, p.precomp_C+1); 

pc_idx = 1;
for i = 1:p.precomp_C
    C_precomp(:,:,i) = fun_basis(s_precomp-ph(i), p.nchan, p.chan_exp);            
    Ctrain(:,:,i) = fun_basis(train_ori-ph(i), p.nchan, p.chan_exp);
end
Ctest = nan(Ntesttrials, p.nchan, p.precomp_C);
if p.prev_C
    Ctrain_prev = vertcat(nan(1,p.nchan, p.precomp_C), Ctrain(1:end-1,:,:));
    train_runNs = p.runNs(p.train_trials);
    sr_train = train_runNs == vertcat(nan, train_runNs(1:end-1));
    Ctrain_prev(~sr_train,:,:) = 0;
    Ctrain = [Ctrain Ctrain_prev];
    test_runNs = p.runNs(p.test_trials);
    sr_test = test_runNs == vertcat(nan, test_runNs(1:end-1));    

    
    test_ori = p.stimval(p.test_trials)/90*pi;
    for i = 1:p.precomp_C
        Ctest(:,:,i) = fun_basis(test_ori-ph(i), p.nchan, p.chan_exp);        
    end
    Ctest_prev = vertcat(nan(1,p.nchan,p.precomp_C), Ctest(1:end-1,:,:));
    Ctest_prev(~sr_test,:,:) = 0; 
end

cnt = zeros(Ntesttrials,100);

%% Find best hyperparameter values (using inner CV-loop within the training data)
if strcmpi(p.dec_type, 'TAFKAP')
    fprintf('\n--PERFORMING HYPERPARAMETER SEARCH--\n')
    lvr = linspace(0,1,50)';
    lr = linspace(0,1,50)'; lr(1) = [];

    hypers = find_lambda(p.runNs(p.train_trials), {lvr, lr});
elseif strcmpi(p.dec_type, 'PRINCE')
    %No hyperparameter search for PRINCE
    p.Nboot=1;
    hypers = [];
else
    error('Invalid decoder type specified');
end


%% Bootstrap loop (run only once for PRINCE)

for i = 1:p.Nboot    
    %% Bootstrap sample of W & covariance
    % Resample train trials with replacement and estimate W and the
    % covariance matrix on this resampled training data. For PRINCE, don't
    % bootstrap, but just estimate W and covariance matrix once for the
    % unresampled training data.
    
    if strcmpi(p.dec_type, 'TAFKAP')
        fprintf('\nBootstrap iteration: %d', i);     
    
        if p.precomp_C>1 
            pc_idx = randi(p.precomp_C-1);        
        else
            pc_idx = 1;
        end
        
        [W, noise] = estimate_W(train_samples, Ctrain(:,:,pc_idx), 1);
        
        cov_est = estimate_cov(noise,hypers(1),hypers(2), W);        
        try
            prec_mat = invChol_mex(cov_est);
        catch ME
            if strcmp(ME.identifier, 'MATLAB:invChol_mex:dpotrf:notposdef')                     
                fprintf('\nWARNING: Covariance estimate wasn''t positive definite. Trying again with another bootstrap sample.\n');
                continue
            else
                rethrow(ME);
            end
        end
        
    else
        fprintf('\n--ESTIMATING PRINCE GENERATIVE MODEL PARAMETERS--\n');
        [W, noise] = estimate_W(train_samples, Ctrain(:,:,1));        
        init_losses = inf;
        while all(isinf(init_losses))
            inits = arrayfun(@(x) rand(Nvox+2,1), (1:20)', 'UniformOutput', 0);
            init_losses = cellfun(@fun_negLL_norm, inits);
        end
        [~, min_idx] = min(init_losses);                
        sol = minimize(inits{min_idx}, @fun_negLL_norm, 1e4);
        prec_mat = invSNC(W(:, 1:pnchan), sol(1:end-2), sol(end-1), sol(end));
    end    
     
    %% Compute likelihoods on test-trials given model parameter sample
    
    pred = C_precomp(:,:,pc_idx)*W(:,1:p.nchan)';        

    if ~mod(i, 100), old_cnt = cnt; end

    for j = 1:Ntesttrials        
        if p.prev_C
            res = test_samples(j,:) - Ctest_prev(j,:,pc_idx)*W(:,p.nchan+1:end)';            
        else
            res = test_samples(j,:);
        end
        res = bsxfun(@minus, res, pred);

        ll = -0.5*MatProdDiag(res*prec_mat, res');

        ps = exp(ll-max(ll)); %For numerical stability, subtract maximum from each log-likelihood
        ps = ps/sum(ps);

        cnt(j,:) = cnt(j,:) + ps';        
    end        

    if ~mod(i, 100) 
        %We check how much the likelihoods have changed after every 100
        %bootstrap samples (not after every sample since the JS-divergence
        %that quantifies the amount of change is somewhat expensive to
        %compute).
        mDJS = max(fun_DJS(bsxfun(@rdivide, old_cnt, sum(old_cnt,2)), bsxfun(@rdivide, cnt, sum(cnt,2))));                
        fprintf('\nChange in likelihoods (JS-divergence) in last 100 iterations: %5.4g\n', mDJS);
        if mDJS < p.DJS_tol, break; end            
    end
    
    
    
end

fprintf \n

liks = bsxfun(@rdivide, cnt, sum(cnt,2)); %(Normalized) likelihoods (= posteriors, assuming a flat prior)
pop_vec = liks*exp(1i*s_precomp); 
est = mod(angle(pop_vec)/pi*90, 180); %Stimulus estimate (likelihood/posterior means)
unc = sqrt(-2*log(abs(pop_vec)))/pi*90; %Uncertainty (defined here as circular SDs of likelihoods/posteriors)
    
    function [minll, minder] = fun_negLL_norm(params)        
        if nargout > 1
            [minll, minder] = fun_LL_norm(params);        
        else 
            minll = fun_LL_norm(params);
        end
        minll = -minll;
        if nargout>1
            minder = -minder;         
        end
    end


    function [loglik, der] = fun_LL_norm(params) 
        
        %Computes the log likelihood of the noise parameters. Also returns 
        %the partial derivatives of each of the parameters (i.e. the 
        %gradient), which are required by minimize.m and other efficient
        %optimization algorithms. 
        
        nvox = size(noise,2);
        ntrials = size(noise,1);
                
        tau = params(1:end-2);
        sig = params(end-1);
        rho = params(end);
        
                        
        [omi, NormConst] = invSNC(W(:,1:p.nchan), tau, sig, rho);
                
        XXt = noise'*noise;
        
        negloglik = 0.5*(MatProdTrace(XXt, omi) + ntrials*NormConst);        
        
        if ~isreal(negloglik), negloglik = Inf; end %If we encounter a degenerate solution (indicated by complex-valued likelihoods), make sure that the likelihood goes to infinity.       
        
        if any(vertcat(tau,sig)<0.001), negloglik = Inf; end
        if abs(rho) > 0.999999, negloglik = Inf; end
        
        loglik = -negloglik;
        
        if nargout > 1
            der = nan(size(params));
                        
            ss = sqrt(ntrials);
            U = (omi*noise')/ss;
            
            dom = omi*(eye(nvox)-((1/ntrials)*XXt)*omi);
            
            JI = 1-eye(nvox);
            R = eye(nvox)*(1-rho) + rho;
            der(1:end-2) = 2*(dom.*R)*tau;
            der(end) = sum(sum(dom.*((tau*tau').*JI)));            
            
            der(end-1) = 2*sig*MatProdTrace(W(:,1:p.nchan).'*omi, W(:,1:p.nchan)) - sum(sum((U.'*sqrt(2*sig)*W(:,1:p.nchan)).^2));

            der = -0.5*ntrials*der;            
            
        end
        
        
    end

    function loss = fun_norm_loss(c_est, c0)
        
        try
            loss = (logdet(c_est, 'chol') + sum(sum(invChol_mex(c_est).*c0)))/size(c0,2);
        catch ME
            if any(strcmpi(ME.identifier, {'MATLAB:posdef', 'MATLAB:invChol_mex:dpotrf:notposdef'}))
                loss = (logdet(c_est) + trace(c_est\c0))/size(c0,2);
            else
                rethrow(ME);
            end
        end
        
    end

    function [W, noise, test_noise] = estimate_W(samples, C, do_boot, test_samples, test_C)        
        if nargin < 3, do_boot = 0; end
        N = size(C,1);
        c_coeff = [];

        
        if do_boot, idx = randi(N,N,1); else, idx = (1:N)'; end
        
        if p.prev_C                         
            W_prev = ([C(idx,p.nchan+1:end) ones(N,1)]\samples(idx,:))';
            W_prev = W_prev(:,1:end-1);                    
            samples = samples - C(:,p.nchan+1:end)*W_prev';
            C = C(:, 1:p.nchan);            
        else
            W_prev = [];
        end
       
        W_curr = (C(idx,:)\samples(idx,:))';                            
        W = [W_curr W_prev];                         
        

        if nargout > 1                        
            noise = samples(idx,:) - C(idx,:)*W_curr';             
            if nargout > 2                 
                test_noise = test_samples - test_C*W';                
            end
        end                   
    end

    function lambda = find_lambda(cvInd, lambda_range)
        % lambda_range can be a vector, in which case the same range is
        % used for the two lambda's. It can also be a matrix with two
        % columns, in which case the first column is used for lambda_var
        % and the second for lambda. Finally, it can be a cell array of
        % two cells, each of which contains a vector, in which case the
        % first vector is used for lambda_var, and the second for lambda
        % (this is the most flexible, as it allows you to specify two
        % different-length ranges for the two hyperparameters). 
        
        if nargin<2, lambda_range = linspace(0,1,50)'; end
                
        cv_folds = unique(cvInd);
        K = length(cv_folds);
        
        assert(K>1, 'Must have at least two CV folds');
        
        clear W_cv est_noise_cv val_noise_cv
        W_cv{K} = []; est_noise_cv{K} = []; val_noise_cv{K} = [];
        
        % Pre-compute tuning weights and noise values to use in each
        % cross-validation split
        for cv_iter=1:K
            val_trials = cvInd==cv_folds(cv_iter);
            est_trials = ~val_trials;            
            est_samples = train_samples(est_trials,:);
            val_samples = train_samples(val_trials,:);
            [W_cv{cv_iter}, est_noise_cv{cv_iter}, val_noise_cv{cv_iter}] = estimate_W(est_samples, Ctrain(est_trials,:,1), 0, val_samples, Ctrain(val_trials,:,1));
        end
        
        % Grid search
        if ~iscell(lambda_range)
            if size(lambda_range,1)<size(lambda_range,2), lambda_range = lambda_range'; end
            if size(lambda_range,2)==1, lambda_range = repmat(lambda_range, 1,2); end        
            lambda_range = mat2cell(lambda_range, size(lambda_range,1), [1 1]);
        end
        
                
        s = cellfun(@length, lambda_range);
        Ngrid = min(max(2, ceil(sqrt(s))), s); %Number of values to visit in each dimension (has to be at least 2, except if there is only 1 value for that dimension)        
        
        grid_vec = cellfun(@(x,y) linspace(1, y, x), num2cell(Ngrid), num2cell(s), 'UniformOutput', 0);        
        [grid_x, grid_y] = meshgrid(grid_vec{1}, grid_vec{2});
        [grid_l1, grid_l2] = meshgrid(lambda_range{1}, lambda_range{2});
        sz = fliplr(cellfun(@numel, lambda_range));
                
        fprintf('\n--GRID SEARCH--');
        losses = nan(numel(grid_x),1);
        for grid_iter=1:numel(grid_x)
            this_lambda = [lambda_range{1}(grid_x(grid_iter)) lambda_range{2}(grid_y(grid_iter))];
            losses(grid_iter) = visit(this_lambda);
            fprintf('\n %02d/%02d -- lambda_var: %3.2f, lambda: %3.2f, loss: %5.4g', [grid_iter, numel(grid_x), this_lambda, losses(grid_iter)]);
        end        
        visited = sub2ind(sz, grid_y, grid_x); visited = visited(:);
        fprintf \n
        
        [best_loss, best_idx] = min(losses);
                        
        % Pattern search
        
        fprintf('\n--PATTERN SEARCH--');        
        step_size = 2^floor(log2(diff(grid_y(1:2)/2))); %Round down to the nearest power of 2 (so we can keep dividing the step size in half)
        while 1            
            [best_y,best_x] = ind2sub(cellfun(@numel, lambda_range), best_idx);
            new_x = best_x + [-1 1 -1 1]'*step_size;
            new_y = best_y + [-1 -1 1 1]'*step_size;
            del_idx = new_x<=0 | new_x> numel(lambda_range{1}) | new_y<=0 | new_y > numel(lambda_range{2});
            new_x(del_idx) = []; new_y(del_idx) = [];
            new_idx = sub2ind(sz, new_y, new_x);
            new_idx = new_idx(~ismember(new_idx, visited));
            if ~isempty(new_idx)
                this_losses = nan(size(new_idx));
                for ii = 1:length(new_idx)
                    this_lambda = [grid_l1(new_idx(ii)), grid_l2(new_idx(ii))];
                    this_losses(ii) = visit(this_lambda);                    
                    fprintf('\nStep size: %d, lambda_var: %3.2f, lambda: %3.2f, loss: %5.4g', [step_size, this_lambda, this_losses(ii)]);
                end                
                visited = vertcat(visited, new_idx);                                    
                losses = vertcat(losses, this_losses);
            end
            
            if any(this_losses<best_loss)
                [best_loss, best_idx] = min(losses);
            elseif step_size>1
                step_size = step_size/2;
            else
                break
            end                
        end
        fprintf \n
        
        lambda = [grid_l1(best_idx), grid_l2(best_idx)];
        
        fprintf('\nBest setting found: lambda_var = %3.2f, lambda = %3.2f, loss = %5.4g\n', [lambda, best_loss]);
        
        function loss = visit(lambda)
            loss = 0;
            for cv_iter2=1:K                
                estC = estimate_cov(est_noise_cv{cv_iter2}, lambda(1), lambda(2), W_cv{cv_iter2});
                valC = (val_noise_cv{cv_iter2}'*val_noise_cv{cv_iter2})/size(val_noise_cv{cv_iter2},1); %sample covariance of validation data
                loss = loss + fun_norm_loss(estC, valC);
            end
            if imag(loss)~=0, loss = inf; end
        end
        
                
    end


    function C = estimate_cov(X,lambda_var,lambda, W)

        [n,pp] = size(X);
        W = W(:,1:p.nchan);

        vars = mean(X.^2);
        medVar = median(vars); 

        t = tril(ones(pp),-1)==1;
        samp_cov = (X'*X/n);

        WWt = W*W';
        coeff = [WWt(t), ones(sum(t(:)),1)]\samp_cov(t);        

        target_diag = lambda_var*medVar + (1-lambda_var)*vars;
        target = coeff(1)*WWt + ones(pp)*coeff(2);
        target(eye(pp)==1)=target_diag;

        C = (1-lambda)*samp_cov + lambda*target;
    end
    

    function out = MatProdTrace(mat1, mat2)
        % Computes the trace of a product of 2 matrices efficiently.
        mat2 = mat2';
        out = mat1(:)'*mat2(:); 
    end   
    

    function out = MatProdDiag(mat1, mat2)        
        % Computes the diagonal of a product of 2 matrices efficiently.        
        out = sum(mat1.*mat2', 2);
    end

    function out = fun_DJS(P,Q)        
        % Computes JS-divergence between corresponding rows in P and Q
        M = P/2+Q/2;
        out = fun_DKL(P,M)/2+fun_DKL(Q,M)/2;        
    end

    function out = fun_DKL(P, Q)
        % Computes KL-divergence from each row in Q to each corresponding
        % row in P
        z = P==0;
        out = P.*(log(P)-log(Q));        
        out(z) = 0;
        out = sum(out,2);        
    end
    
            
end
