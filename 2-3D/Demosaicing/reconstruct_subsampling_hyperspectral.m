
% Parameters
lambda_residual = 100000;
lambda = 1.0; 
verbose_admm = 'brief';
max_it = 200;

% Load hyperspectral data as variable b 
% Format should be [x, y, wavelengths, indexes]
% For information on finding hyperspectral data please see the readme
% For best results all data should be similarly normalized but not contrast
% normalized.
load('data_to_demosaic.mat', 'b');
b_all = b;

% Load Dictionary
load('../Filters/2D-3D-Hyperspectral.mat', 'd');

% Sampling matrix
% This pattern follows the sampling scheme illustrated in the supplemental material.
sb = sqrt(size(b,3));
MtM = zeros(size(b_all(:,:,:,1)));
[I,J] = meshgrid(1:sb:size(b,1));
c = 1;
for m=1:sb
for n=1:sb
    MtM(I+(m-1),J+(n-1),c) = 1;
    c = c+1;
end
end

% Iterate over all hyperspectral images
for i=1:size(b_all, 4)
    
b = b_all(:,:,:,i);

% Choose the current image and remove all of the unkown data
signal = b(:,:,:);
signal( ~MtM ) = 0;
signal_sparse = b(:,:,:);
signal_sparse( ~MtM ) = 0;

% Nearest neighbour filling
% For the purpose of local contrast normalization we need a rough estimate
% fo the data at all points so we choose to fill them with NN
for w=1:size(b,3)
    a = signal(:,:,w);
    [~, idx] = bwdist(a ~= 0);
    a(a == 0) = a(idx(a == 0));
    signal(:,:,w) = a;
end

% Filter from local contrast normalization
k = fspecial('gaussian',[13 13],3*1.591); 
smooth_init = imfilter(reshape(signal, size(b)), k, 'same', 'conv', 'symmetric');

fprintf('Doing sparse coding reconstruction.\n\n')
tic();
[z, sig_rec] = admm_solve_conv23D_weighted_sampling(signal_sparse, d, MtM, lambda_residual, ...
                                            lambda, max_it, 1e-3, [], verbose_admm, smooth_init); 
tt = toc;
fprintf('Done sparse coding! --> Time %2.2f sec.\n\n', tt)

end


