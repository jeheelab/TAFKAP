function c = fun_basis(s, nchan, expo)
    % Basis functions for fitting (orientation) tuning functions.     
    if nargin < 3, expo = 5; end
    if nargin < 2, nchan = 8; end
    TuningCentres = 0:2*pi/nchan:2*pi-0.001; 
    assert(size(s,2)==1, 's must be a scalar or column vector');        
    c = max(0, cos(bsxfun(@minus, s, TuningCentres))).^expo;        
end