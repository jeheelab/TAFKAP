function c = fun_basis(s, nchan, expo, classes)
    % Basis functions for fitting (orientation) tuning functions.     
    if nargin < 4, classes = []; end
    if nargin < 3, expo = []; end
    if nargin < 2, nchan = []; end
        
    if isempty(expo), expo=5; end
    if isempty(nchan), nchan=8; end
    
    assert(isvector(s), 'Input variable ''s'' must be a vector');
    d = size(s);
    if d(2)>d(1), s=s'; end %Make sure s is a column vector
    
    if ~isempty(classes)
        assert(isvector(classes), 'Input variable ''classes'' must be a vector');
        d = size(classes);        
        if d(1)>d(2), classes=classes'; end %Make sure classes is a row vector
        c = s==classes;
    else
        TuningCentres = 0:2*pi/nchan:2*pi-0.001;     
        c = max(0, cos(bsxfun(@minus, s, TuningCentres))).^expo;        
    end
end