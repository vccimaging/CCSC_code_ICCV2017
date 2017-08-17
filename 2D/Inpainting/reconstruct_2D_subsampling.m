% Numerical test for 2D mixture model fitting using sparse coding

clear;
close all;

%% Debug options
verbose = 'all';

addpath('../../image_helpers');
CONTRAST_NORMALIZE = 'none'; 
ZERO_MEAN = 0;   
COLOR_IMAGES = 'gray';   
[b] = CreateImages('./Test',CONTRAST_NORMALIZE,ZERO_MEAN,COLOR_IMAGES);
signal = b(:,:,:,2);


% % %Sampling matrix
MtM = ones(size(signal));
% % %MtM(1:2:end, 1:2:end) = 1;
MtM(rand(size(MtM)) < 0.5 ) = 1;
signal_sparse = signal;
signal_sparse( ~MtM ) = 0;


%% Load filters
%kernels = load('./init_results/filters_bristow_obj5.4e+04_11:06:19:29.mat');
%kernels = load('filters_ours_obj1.26e+04_11:07:15:13.mat');
kernels = load('Filters_ours_2D_large.mat');

d = kernels.d;

%Show kernels
if strcmp(verbose, 'brief ') || strcmp(verbose, 'all') 
    figure();
    sqr_k = ceil(sqrt(size(d,3))); pd = 1;
    psf_radius = floor(size(d,1)/2);
    d_disp = zeros( sqr_k * [psf_radius*2+1 + pd, psf_radius*2+1 + pd] + [pd, pd]);
    for j = 0:size(d,3) - 1
        d_disp( floor(j/sqr_k) * (size(d,1) + pd) + pd + (1:size(d,1)) , mod(j,sqr_k) * (size(d,2) + pd) + pd + (1:size(d,2)) ) =  d(:,:,j + 1);
    end
    imagesc(d_disp), colormap gray, axis image, colorbar, title('Kernels used');
    
    figure();
    subplot(1,2,1), imagesc( signal ), axis image, colormap gray, title('Original image');
    subplot(1,2,2), imagesc( signal_sparse ), axis image, colormap gray, title('Subsampled image');
end

%% 1) Sparse coding reconstruction     
fprintf('Doing sparse coding reconstruction.\n\n')

lambda_residual = 5.0;
lambda = 2.0; %

verbose_admm = 'all';
max_it = 100;
tic();
[z, sig_rec] = admm_solve_conv2D_weighted_sampling(signal_sparse, d, MtM, lambda_residual, lambda, max_it, 1e-3, signal, verbose_admm); 
tt = toc;

%Show result
if strcmp(verbose, 'brief ') || strcmp(verbose, 'all') 
    figure();
    subplot(1,2,1), imagesc(signal), axis image, colormap gray; title('Orig');
    subplot(1,2,2), imagesc(sig_rec), axis image, colormap gray; title('Reconstruction');
end

%Debug
fprintf('Done sparse coding! --> Time %2.2f sec.\n\n', tt)

%Write stuff
max_sig = max(signal(:));
min_sig = min(signal(:));

%Transform and save
signal_disp = (signal - min_sig)/(max_sig - min_sig);
signal_sparse_disp = (signal_sparse - min_sig)/(max_sig - min_sig);
signal_sparse_disp( ~MtM ) = 0;
sig_rec_disp = (sig_rec - min_sig)/(max_sig - min_sig);

max_d = max(d_disp(:));
min_d = min(d_disp(:));
d_sc = (d - min_d)/(max_d - min_d);

sqr_k = ceil(sqrt(size(d,3))); pd = 1;
psf_radius = floor(size(d,1)/2);
d_disp = ones( sqr_k * [psf_radius*2+1 + pd, psf_radius*2+1 + pd] + [pd, pd]);
for j = 0:size(d,3) - 1
    d_disp( floor(j/sqr_k) * (size(d,1) + pd) + pd + (1:size(d,1)) , mod(j,sqr_k) * (size(d,2) + pd) + pd + (1:size(d,2)) ) =  d_sc(:,:,j + 1);
end

%Save stuff
imwrite(signal_disp , 'signal.png','bitdepth', 16);
imwrite(signal_sparse_disp ,'signal_sparse.png','bitdepth', 16);
imwrite(sig_rec_disp ,'signal_reconstruction.png','bitdepth', 16);
imwrite(d_disp ,'kernel.png','bitdepth', 16);
