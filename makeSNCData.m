function [rsp, simpar] = makeSNCData(p)

% Simulates fMRI voxel activations to oriented stimuli across a number of
% trials. Voxel responses contain noise that is spatially correlated, esp.
% between voxels with similar orientation tuning preferences. For added
% realism, we simulate data in discrete fMRI runs. 

def = {
    'randseed', 1234;
    'Wstd', 0.3;
    'rho', 0.05;
    'sigma', 0.3;
    'nvox', 100;
    'taumean', 0.7;
    'taustd', 0.035;
    'nchan', 8;
    'nclasses', 4;
    'ntrials', 220;
    'ntrials_per_run', 20;
    'sigma_arb', 0.3;    
    'sim_stim_type', 'circular'; 
    };

if nargin < 1, p = struct; end

p = setdefaults(def, p);

try
    rng(p.randseed, 'twister');    
catch
    rs = RandStream('mt19937ar', 'Seed', p.randseed); 
    RandStream.setDefaultStream(rs);    
end

nruns = p.ntrials/p.ntrials_per_run;
assert(round(nruns)==nruns, 'Number of trials per run must divide evenly into total number of trials');
run_idx = sort(repmat((1:nruns)', p.ntrials_per_run, 1));

switch p.sim_stim_type
    case 'circular'
        %This code generates orientations by (roughly) the same pseudorandom 
        %procedure used in our fMRI experiments, ensuring that each run uniformly
        %samples orientation space without any biases. 
        base_ori = linspace(0, 2*pi, p.ntrials_per_run+1)'; base_ori(end)=[];
        run_offsets = rand(1,nruns)*diff(base_ori(1:2));
        ori = bsxfun(@plus, base_ori, run_offsets);
        for j = 1:size(ori,2)
            [~, idx] = sort(rand(size(ori,1),1));
            ori(:,j) = ori(idx,j);
        end
        ori = ori(:);
        stimval = ori;
        classes = [];
    case 'categorical'
        %This code generates balanced runs of categorical stimuli
        p.nchan = p.nclasses;
        assert(mod(p.ntrials_per_run, p.nclasses)==0, 'To simulate categorical stimulus labels, number of classes must divide neatly into the number of trials per run.');                
        tmp = repmat((1:p.nclasses)', p.ntrials_per_run/p.nclasses, 1);
        [~, orders] = arrayfun(@(x) sort(rand(x,1)), ones(1,nruns)*p.ntrials_per_run, 'UniformOutput', false);
        orders = horzcat(orders{:});
        stimval = tmp(orders(:)); 
        classes = 1:p.nclasses;
end


%Simulate generative model parameters:
W = randn(p.nvox,p.nchan)*p.Wstd;                       
tau_sim = randn(p.nvox,1)*p.taustd+p.taumean;           
sig_sim = p.sigma;                                          
rho_sim = p.rho;

[~, order] = sort(rand(p.nvox,1));
W_shuffled = W(order,:);
cov_sim = (1-rho_sim)*diag(tau_sim.^2) + rho_sim*(tau_sim*tau_sim') + sig_sim^2*(W*W') + p.sigma_arb^2*(W_shuffled*W_shuffled'); %See van Bergen & Jehee (2018), Neuroimage

%Simulate data:
Q = chol(cov_sim, 'lower'); %Mixing matrix for generating correlated noise (this transforms standard IID Normal noise to multivariate Normal noise with covariance structure cov_sim)
noise = (Q*randn(p.ntrials,p.nvox)')'; %(Note that the sample covariance of this noise will not exactly equal cov_sim. cov_sim is the covariance matrix for the population from which the sample is drawn.)          
tun = fun_basis(stimval, [], [], classes)*W'; %Tuning curves sampled at the simulated stimulus orientations
rsp = tun + noise; %Simulated voxel responses 

if nargout > 1
    simpar = p;
    simpar.W = W;
    simpar.stimval = stimval;
    simpar.tau = tau_sim;
    simpar.rho = rho_sim;
    simpar.sigma = sig_sim;        
    simpar.sigma_arb = p.sigma_arb;
    simpar.prec_mat = inv(cov_sim);
    simpar.run_idx = run_idx;
end

end