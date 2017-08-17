
clear;
close all;

% Parameters
lambda_residual = 10000.0;
lambda = 1/8; 
verbose_admm = 'brief';
max_it = 120;
tol = 1e-6;

% Load video data as variable 'b'
% Format should be [x, y, time, indexes]
% For information on finding video data please see the readme
% For best results all data should be similarly normalized but not contrast
% normalized.
load('videos/testing_data.mat', 'b');
b_all = b(:,:,:,:);

% Load Dictionary as variable 'd'
load('../filters/3D_video_filters.mat', 'd');

% Sampling matrix
% For this experiment we use all of the data so MtM is all ones
MtM = ones(size(b_all(:,:,:,1)));

% Load psf to convolve with video data
psf = imread('snake.png');
psf = imresize(double(psf(:,:,1)), [3,3]);
psf = psf./sum(psf(:));
psf = repmat(psf,[1,1,3]);
psf(:,:,1) = 0;
psf(:,:,3) = 0;

% Iterate over all video clips
for i=1:size(b_all,4)

    % Convolve the video
    b_clean = mat2gray(b_all(:,:,:,i));
    b = imfilter(b_clean, psf, 'same', 'conv', 'symmetric');

    % Per-Frame Mean Normalization
    vmsi = permute(reshape(double(b), [], size(b,3)), [2 1]);
    veam = mean(vmsi,2);
    vstd = std(vmsi, 0, 2);
    vmsi = (vmsi-repmat(veam, [1 size(vmsi,2)])) ./ repmat(vstd, [1 size(vmsi,2)]);
    nb = reshape(vmsi', size(b));

    % Filter from local contrast normalization
    k = fspecial('gaussian',[15 15],3 * 1.591); 
    smooth_init = imfilter(nb, k, 'same', 'conv', 'symmetric');

    fprintf('Doing sparse coding reconstruction.\n\n')
    tic();

    [z, Dz]  = admm_solve_video_weighted_sampling(nb, d, MtM, lambda_residual, lambda, max_it, tol, verbose_admm, psf, smooth_init);

    tt = toc;
    fprintf('Done sparse coding! --> Time %2.2f sec.\n\n', tt)

    sig_rec = Dz;

    % Un-normalize
    vemm = reshape(permute(repmat(veam, [1 size(sig_rec,2)*size(sig_rec,1)]), [2 1]), size(sig_rec));
    vssd = reshape(permute(repmat(vstd, [1 size(sig_rec,2)*size(sig_rec,1)]), [2 1]), size(sig_rec));
    
    % This is the final output
    sig_rec_disp = (sig_rec .* vssd + vemm);

end

