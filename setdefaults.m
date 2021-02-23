function s = setdefaults(defaults, s)

% Sets default values for fields in struct 's'. Defaults are
% specified as key-value pairs in rows of the cell array 'defaults'. Fields
% already present in 's' are left as they are, also when their value 
% is empty. Fields that are mentioned in 'defaults' but not set in 's' are
% set to the values in 'defaults'.
%
%
% Example:
%
% p = struct('Iterations', 10, 'Smoothing', 2);
% defaults = {'Iterations', 8; 'Smoothing_kernel', 'Gaussian'; ...
%   'Smoothing', 2};
% p = setdefaults(defaults, p)
%
%   disp(p)
%               Iterations: 10
%                Smoothing: 2
%         Smoothing_kernel: 'Gaussian'
%
% Note that only the 'Smoothing_kernel' field, which hadn't been assigned
% in 'p', was set to its default value. Note also that field order is not
% changed.

if nargin < 2, s = []; end
if isempty(s), s = struct; end

for f_idx = 1:size(defaults,1)
    if ~isfield(s, defaults{f_idx, 1}) || isempty(s.(defaults{f_idx, 1}))
        s.(defaults{f_idx, 1}) = defaults{f_idx, 2};
    end    
end

end