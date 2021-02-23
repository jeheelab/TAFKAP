function [omi, ld] = invSNC(W, tau, sig, rho)

% This code implements an efficient way of computing the inverse and log 
% determinant of the covariance matrix used in van Bergen, Ma, Pratte & 
% Jehee (2015), in the form of two low-rank updates of a diagonal matrix.
%
% Reference:
% van Bergen, R. S., Ma, W. J., Pratte, M. S., & Jehee, J. F. M. (2015). 
% Sensory uncertainty decoded from visual cortex predicts behavior. 
% Nature neuroscience, 18(12), 1728-1730. https://doi.org/10.1038/nn.4150

nvox = size(W,1);

if sig==0 && rho==0 
    omi = diag(tau.^-2);
    ld = 2*sum(log(tau));        
else
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
    
    
    alpha = 1/(1-rho);
    Fi = alpha*sparse(diag(tau.^-2));            
    ti = 1./tau;
    Di = Fi - (rho*alpha^2*(ti*ti'))/(1+rho*nvox*alpha); %Sherman-Morrison

    DiW = Di*W;
    WtDiW = W'*DiW;
    omi = Di - DiW/(sig^-2*eye(size(W,2))+WtDiW)*DiW'; %Woodbury

    if nargout > 1
        % We can apply similar tricks to the log determinant. In this case,
        % this is based on the Matrix Determinant Lemma.
        % (https://en.wikipedia.org/wiki/Matrix_determinant_lemma)
        try
            ld = logdet(sparse(eye(size(W,2)))+sig^2*WtDiW, 'chol') + log(1 + rho*nvox*alpha) + nvox*log(1-rho) + 2*sum(log(tau));  
        catch ME        
            if strcmpi(ME.identifier, 'MATLAB:posdef')
                % If the cholesky decomposition-based version of logdet 
                % failed, this may be an indication that your optimization 
                % routine is searching in the wrong region of the solution 
                % space. If the optimization finishes shortly after seeing
                % this message, you shouldn't trust the result. This may be 
                % due to a bad initialization of the noise parameters.
                warning('Cholesky decomposition-based computation of log determinant failed. Trying with LU decomposition instead.');
                ld = logdet(sparse(eye(size(W,2)))+sig^2*WtDiW) + log(1 + rho*nvox*alpha) + nvox*log(1-rho) + 2*sum(log(tau));  
            else
                % If the failure is not due to a violation of 
                % positive-definiteness, rethrow the exception.
                rethrow(ME);
            end
        end
       
    end
end


end