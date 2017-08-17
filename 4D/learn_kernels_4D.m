% Learning test for sparse convolutional coding

clear;
close all;

%% Debug options
verbose = 'brief';

%Load lightfield dataset
fn = './Datasets_lf/food_localCN_bis3_8x8.mat';
I = load(fn);
b = I.b;
    
size_I = size(b);
ndims = length(size_I) - 1;
n_vid = size_I(end);

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

    %Iterate over spec
    figure();     
    for k2 = 1:size(b,4)
      for k1 = 1:size(b,3)
          %for j = 1:size(b,3)
            imshow(b_view(:,:,k1, k2),[]);
            title(sprintf('Extracted Multispec [ VIEW (%d, %d) ]',k1, k2));
            pause(0.01);
          %end
      end
    end
    
    %Iterate over views
    figure();     
    %for j = 1:size(b,3)     
       for k2 = 1:size(b,4)
         for k1 = 1:size(b,3)
            imshow(b_view(:,:,k1, k2),[]);
            title(sprintf('Extracted Multispec [ VIEW (%d, %d)]',k1, k2));
            pause(0.01);
          end
      end
    %end
    
end

%% Define the parameters
kernel_size = [11, 11, 5, 5, 49];
lambda_residual = 1.0;
lambda = 1.0; %2.8


%% Do the reconstruction  
fprintf('Doing sparse coding kernel learning for k = %d [%d x %d x %d x %d] kernels.\n\n', kernel_size(5), kernel_size(1), kernel_size(2), kernel_size(3), kernel_size(4) )

%Optim options
verbose_admm = 'all';
max_it = [20]; %[300];
tol = 0.001;

tic();

prefix = 'ours';
[ d, z, Dz, obj, iterations]  = admm_learn_conv4D_lightfield(b, kernel_size, lambda_residual, lambda, max_it, tol, verbose_admm, []);

tt = toc

%Show result
k  = 49; %kernel_size(end);
ndims = 4;
psf_radius = 5;
sqr_k = ceil(sqrt(k));
pd = 1;
d_disp = zeros( sqr_k * [psf_radius*2+1 + pd, psf_radius*2+1 + pd] + [pd, pd]);
inds = repmat({3}, 1, ndims + 1);
inds{1} = ':'; %Pick first two dims to show in 2D window
inds{2} = ':';
for j = 0:k - 1
    inds{end} = j + 1;
    d_curr = d(inds{:}); 
    d_curr = d_curr(1:psf_radius*2+1, 1:psf_radius*2+1);
    d_disp( floor(j/sqr_k) * (size(d_curr,1) + pd) + pd + (1:size(d_curr,1)) , mod(j,sqr_k) * (size(d_curr,2) + pd) + pd + (1:size(d_curr,2)) ) = d_curr;
end
figure;imagesc(d_disp), colormap gray, axis image, colorbar;
drawnow;

%Debug
fprintf('Done sparse coding learning >>> saving.\n\n', tt)

%Save
%%dt = datestr(now,'mm:dd:HH:MM');
save(sprintf('Filters_lightfield_%s.mat', prefix), 'd');

%Debug
fprintf('Done sparse coding learning! --> Time %2.2f sec.\n\n', tt)
