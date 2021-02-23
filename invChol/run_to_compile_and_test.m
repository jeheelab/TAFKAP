% Description:
%       Computes the inverse of a symmetric matrix using the cholesky
%       decomposition and compares the runtime to that of the standard inv
%       function.
%
% Author: 
%		Eric Blake
%		Black River Systems Company
%		blake@brsc.com
%		05/14/2012


% Compile the MEX file.
clc;
disp('Compiling invChol_mex.c...');
if strcmpi(computer, 'PCWIN64') || strcmpi(computer, 'GLNXA64')
    disp('64-bit detected, compiling with -largeArrayDims flag...');
    mex invChol_mex.c -lmwlapack -largeArrayDims;
else
    disp('32-bit detected, compiling without -largeArrayDims flag...');
    mex invChol_mex.c -lmwlapack;
end

% Number of multiplies to test.
n_mults = 100000;

% Size of matrix to test.
matdim = 10;

% Generate random symmetric matrix.
disp(['Generating random ' num2str(matdim) ' x ' num2str(matdim) ...
    ' symmetric matrix....']);

A=full(sprandsym(matdim,1, 0.5, 1));

% Compute the inverse n_mults times using standard inv functin (uses LU).
disp('Executing traditional MATLAB inv...');
drawnow; % Ensures previous line is executed.
tic;
for i=1:n_mults
    inv(A);
end
t_standard=toc;
disp(['Runtime: ' num2str(t_standard) ' seconds']);

% Compute the inverse n_mults times using cholesky inverse.
A_cholinv = invChol_mex(A); % Run one time first to catch any bugs.
disp('Executing invChol_mex...');
drawnow; % Ensures previous line is executed.
tic;
for i=1:n_mults
    invChol_mex(A);
end
t_chol=toc;
disp(['Runtime: ' num2str(t_chol) ' seconds']);

% Compute and display the max numerical difference.
err = A_cholinv - inv(A);
str =sprintf('Maximum numerical difference between inv and invChol_mex: %d', ...
    max((err(:))));
disp(str);




