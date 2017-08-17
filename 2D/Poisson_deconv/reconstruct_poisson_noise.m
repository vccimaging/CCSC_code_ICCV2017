clear;
close all;

verbose = 'all';

%% Dataset definition
outputfolder = 'results_rec_dataset';
mkdir( outputfolder );

%Load the images
addpath('../../image_helpers');
CONTRAST_NORMALIZE = 'none'; 
ZERO_MEAN = 0;   
COLOR_IMAGES = 'gray';   
[b] = CreateImagesList('./dataset_norm',CONTRAST_NORMALIZE,ZERO_MEAN,COLOR_IMAGES);
nfiles = length(b); 

%Save stuff
PSNRvals = zeros(2, nfiles);
MSEvals = zeros(2, nfiles);
timings = zeros(2, nfiles);

%Iterate over the files
for ii=1:nfiles
    
    clearvars -except verbose outputfolder b nfiles ii PSNRvals MSEvals timings

    fprintf('##################################################\n')
    fprintf('PROCESSING IMAGE [%d/%d] \n', ii, nfiles)
    fprintf('##################################################\n\n')
    
    %% Load image
    signal = double(b{ii});
    % Sampling Matrix
    rate = 1;
    M = rand(size(signal));
    M(rand(size(M)) < rate ) = 1;
    signal_sparse = signal .* M;

    %Poisson Noise
    lmin = 1;
    lmax = 1000;
    signal_sparse = floor( rescale(signal_sparse,lmin,lmax) );
    signal_sparse = ( poissrnd(signal_sparse) - lmin) / (lmax - lmin);
    
    % % Gaussian Noise
    %%signal_sparse = imnoise(signal_sparse, 'gaussian');
    
    % Normalization
% %     vmsi = permute(reshape(signal_sparse, [], 1), [2 1]);
% %     veam = mean(vmsi,2);
% %     vstd = std(vmsi, 0, 2);
% %     vmsi = (vmsi-repmat(veam, [1 size(vmsi,2)])) ./ repmat(vstd, [1 size(vmsi,2)]);
% %     signal_sparse = reshape(vmsi', size(signal_sparse));

% %     % Contrast Normalized b
% %     smooth_init = reshape(Interpolation_Initial(signal_sparse * 255,~M) / 255, size(signal));
% %     k = fspecial('gaussian',[13 13],3*1.591); %Filter from local contrast normalization
% %     smooth_init = imfilter(smooth_init, k, 'same', 'conv', 'symmetric');    
    
    kernels = load('Filters_ours_2D_large.mat');
    d = kernels.d;

    %Show kernels
    if strcmp(verbose, 'all') 
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
        drawnow;
    end

    %% 1) Sparse coding reconstruction     
    fprintf('Doing sparse coding reconstruction.\n\n')    
    
    
    % Parameters
    lambda_residual = 20000.0;
    lambda = 1.0; 
    verbose_admm = 'all';
    max_it = 50;


    % Reconstruction with Poisson Prior
    [z, sig_rec] = admm_solve_conv_poisson(signal_sparse, d, M, lambda_residual, lambda, max_it, 1e-3, signal, verbose_admm); 

    % Reconstruction without Poisson Prior
    %[zold, old_rec] = admm_solve_conv23D_weighted_sampling(signal_sparse, d, M, lambda_residual, lambda, max_it, 1e-3, signal, verbose_admm, smooth_init); 

    % Re-Normalization
    vemm = reshape(permute(repmat(veam, [1 size(sig_rec,2)*size(sig_rec,1)]), [2 1]), size(sig_rec));
    vssd = reshape(permute(repmat(vstd, [1 size(sig_rec,2)*size(sig_rec,1)]), [2 1]), size(sig_rec));
    sig_rec_disp = (sig_rec .* vssd + vemm);
    old_rec_disp = (old_rec .* vssd + vemm);

    % Inspect Results
    psnr(mat2gray(b), mat2gray(sig_rec_disp))
    psnr(mat2gray(b), mat2gray(old_rec_disp))
end

