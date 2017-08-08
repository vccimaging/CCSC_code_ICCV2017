
% Define dictionary parameters [x, y, wavelengths, number of filters]
% For lightfields, etc. you may want e.g., [x, y, sx, sy, n.o.f]
kernel_size = [11, 11, 31, 100];
lambda_residual = 1.0;
lambda = 1.0; 
verbose = 'all';

% Load many 31 channel hyperspectral images
load('learning_data.mat', 'b');

% Filter from local contrast normalization
k = fspecial('gaussian',[13 13],3*1.591); 
smooth_init = imfilter(b, k, 'same', 'conv', 'symmetric');

% Reconstruct & learn filters
fprintf('Doing sparse coding kernel learning for k = %d [%d x %d] kernels.\n\n', kernel_size(3), kernel_size(1), kernel_size(2) )

% Optimization options
verbose_admm = 'brief';
max_it = 40;
tol = 1e-3;
init = [];

% Run optimization
tic();
[d, z, Dz, obj]  = admm_learn(b, kernel_size, lambda_residual, lambda, max_it, tol, verbose_admm, init, smooth_init);
tt = toc;

% Save dictionaryh
save('hyperspectral_dictionary.mat', 'd', 'z', 'Dz', '-v7.3');

% Debug
fprintf('Done dictionary learning! --> Time %2.2f sec.\n\n', tt)

