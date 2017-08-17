% Learning test for sparse convolutional coding

clear;
close all;

%% Debug options
verbose = 'all';

%% Load the movie 
    
% Load contrast normalized video data (created by extractContrastNormalizatonMovie.m and extractMovie.m)
I = load('./movs/drift_hungary_1280x720/full_movie_localCN.mat'); 


n_vid = 8 * 8;
s_data = 50;

%% Show the full movie
%{
if strcmp(verbose, 'all') 
  figure();
  for f = 1:1:size(I,3)
    imshow(I(:,:,f),[]);
    title(sprintf('Movie Frame %03d', f));
    pause(0.01);
  end
end
%}

%Size
size_I = size(I);

%Extract the individual movies
ndims = length(size_I);
offsets_movs = ceil( rand(ndims, n_vid) .* repmat( size_I(:) - (s_data + 1), 1, n_vid ) );

%Sample
b = eval(['zeros(' repmat('s_data,',1,ndims), 'n_vid)']);
for i = 1:n_vid
    offset_start = offsets_movs(:,i);
    range_str = sprintf( repmat('%d + (1:s_data),',1,ndims), offset_start );
    eval(['curr_mov = I(' , range_str(1:end-1), ');']);
    eval(['b(' repmat(':,',1,ndims), 'i) = curr_mov;']);
end

%Show extracted videos
if strcmp(verbose, 'all')
    
    n_view = sqrt( n_vid );
    b_view = [];
    for r = 1:n_view
        b_r = [];
        for c = 1:n_view
            i = (r - 1) * n_view + c;
            b_curr = eval(['b(' repmat(':,',1,ndims), 'i)'] );
            b_curr = eval(['padarray(b_curr,[1,1', repmat(',0',1,ndims - 2), '],0,''both'')']);
            b_r = cat(2, b_r, b_curr );
        end
        b_view = cat(1, b_view,  b_r );
    end

    figure();
    for f = 1:size(b_view,3)
        imshow(b_view(:,:,f),[]);
        title(sprintf('Extracted Movies Frame %03d', f));
        pause(0.01);
    end
end

%% Define the parameters
kernel_size = [11, 11, 11, 49];
lambda_residual = 1.0;
lambda = 1.0; %2.8


%% Do the reconstruction  
fprintf('Doing sparse coding kernel learning for k = %d [%d x %d x %d] kernels.\n\n', kernel_size(4), kernel_size(1), kernel_size(2), kernel_size(3) )

%Optim options
verbose_admm = 'all';
max_it = 20; %[300];
tol = 1e-2;

tic();
[ d, z, DZ, obj, iterations]  = admm_learn_conv3D_large(b, kernel_size, lambda_residual, lambda, max_it, tol, verbose_admm, []);
tt = toc

%Debug
fprintf('Done sparse coding learning >>> saving.\n\n')

k  = 49;
psf_radius = 5;
ndim = 3;
sqr_k = ceil(sqrt(k));
pd = 1;
d_disp = zeros( sqr_k * [psf_radius*2+1 + pd, psf_radius*2+1 + pd] + [pd, pd]);
inds = repmat({5}, 1, ndim + 1);
inds{1} = ':'; %Pick first two dims to show in 2D window
inds{2} = ':';
for j = 0:k - 1
    inds{end} = j + 1;
    d_curr = circshift( d(inds{:}), [psf_radius, psf_radius] ); 
    d_curr = d_curr(1:psf_radius*2+1, 1:psf_radius*2+1);
    d_disp( floor(j/sqr_k) * (size(d_curr,1) + pd) + pd + (1:size(d_curr,1)) , mod(j,sqr_k) * (size(d_curr,2) + pd) + pd + (1:size(d_curr,2)) ) = d_curr;
end
figure;imagesc(d_disp), colormap gray, axis image, colorbar;
drawnow;

%Save
prefix = 'ours';
save(sprintf('Filters_%s_videos.mat', prefix), 'd', 'DZ', 'obj', 'iterations');
