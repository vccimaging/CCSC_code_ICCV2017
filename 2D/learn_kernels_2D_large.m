% Learning test for sparse convolutional coding

clear;
close all;

%% Load the data

addpath('./image_helpers');
CONTRAST_NORMALIZE = 'local_cn'; 
ZERO_MEAN = 1;   
COLOR_IMAGES = 'gray';                         
[b] = CreateImages('./Large_Datset/',CONTRAST_NORMALIZE,ZERO_MEAN,COLOR_IMAGES);

%One long dataset iterating over color if defined
b = reshape(b, size(b,1), size(b,2), [] ); 

%% Define the parameters
kernel_size = [11, 11, 100];
lambda_residual = 1.0;
lambda = 1.0; %2.8


%% Do the reconstruction  
fprintf('Doing sparse coding kernel learning for k = %d [%d x %d] kernels.\n\n', kernel_size(3), kernel_size(1), kernel_size(2) )

%Optim options
verbose_admm = 'all';
max_it = 20;
tol = 1e-3;

tic();

prefix = 'ours';
[ d, z, Dz, iterations]  = admm_learn_conv2D_large(b, kernel_size, lambda_residual, lambda, max_it, tol, verbose_admm, []);

tt = toc

%Show result
psf_radius = 5;
figure();    
pd = 1;
sqr_k = ceil(sqrt(size(d,3)));
d_disp = zeros( sqr_k * [kernel_size(1) + pd, kernel_size(2) + pd] + [pd, pd]);
for j = 0:size(d,3) - 1
    d_disp( floor(j/sqr_k) * (kernel_size(1) + pd) + pd + (1:kernel_size(1)) , mod(j,sqr_k) * (kernel_size(2) + pd) + pd + (1:kernel_size(2)) ) = d(:,:,j + 1); 
end
imagesc(d_disp), colormap gray, axis image, colorbar, title('Final filter estimate');


%Save
save(sprintf('Filters_%s_2D_large.mat', prefix), 'd', 'Dz', 'iterations');

%Debug
fprintf('Done sparse coding learning! --> Time %2.2f sec.\n\n', tt)
