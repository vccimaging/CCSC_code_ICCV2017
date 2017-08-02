function [ d_res, z_res, DZ, obj_val, iterations ] = admm_learn_convND_large(b, kernel_size, ...
                    lambda_residual, lambda_prior, ...
                    max_it, tol, ...
                    verbose, init)
    
    %Kernel size contains kernel_size = [psf_s, psf_s, k]
    psf_s = kernel_size(1);
    k = kernel_size(end);
    sb = size(b);
    n = sb(end);
    ni=sqrt(n);        
    N = n/ni;
                
    %PSF estimation
    psf_radius = floor( psf_s/2 );
    size_x = [sb(1:end - 1) + 2*psf_radius, n];
    size_z = [size_x(1:end - 1), k, n];
    size_z_crop = [size_x(1:end - 1), k, ni];
    size_d_full = [size_x(1:end - 1), k]; 
     
    lambda = [lambda_residual, lambda_prior];

    B = padarray(b, [psf_radius, psf_radius, psf_radius, 0], 0, 'both');
    for i=1:n
        B_hat(:,:,:,i) = fftn(B(:,:,:,i));
    end
    for nn=1:N
        Bh{nn} = B_hat(:,:,:,(nn-1)*ni + 1:nn*ni) ;
    end 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Proximal Operators
    %ProxDataMasked = @(u, B, theta) (B + 1/theta * u ) ./ ( 1 + 1/theta ); 
    ProxSparse = @(u, theta) max( 0, 1 - theta./ abs(u) ) .* u; 
    ProxKernelConstraint = @(u) KernelConstraintProj( u, size_d_full, psf_radius);     
    % Objective
    objective = @(z, d) objectiveFunction( z, d, b, lambda_residual, lambda_prior, psf_radius, size_z, size_x );      
 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    d = padarray( randn(kernel_size), [size_x(1) - kernel_size(1), size_x(2) - kernel_size(2), size_x(3) - kernel_size(3),0], 0, 'post');
    d = circshift(d, -[psf_radius, psf_radius, psf_radius, 0] );
    D = repmat({d}, N,1);
    d_hat = zeros(size(d));
    for ik = 1:k
        d_hat(:,:,:,ik) = fftn( d(:,:,:,ik) );
    end
    D_hat = repmat({d_hat},N,1);
        
    z = randn(size_z);
    z_hat = zeros(size(z));
    %Z = repmat({z}, N,1);
    for in = 1:n
        for ik = 1:k
            z_hat(:,:,:,ik, in) = fftn( z(:,:,:,ik,in) );
        end
    end
    %Z_hat = repmat({z_hat}, N,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
        iterate_fig = figure();
        filter_fig = figure();
        display_func(iterate_fig, filter_fig, d, z_hat, b, size_x, size_z_crop, psf_radius, 0);
    end
    if strcmp( verbose, 'brief') || strcmp(verbose, 'all')
        obj_val = objective(z, d);
        fprintf('Iter %d, Obj %3.3g, Diff %5.5g\n', 0, obj_val, 0)
        obj_val_filter = obj_val;
        obj_val_z = obj_val;
    end
    
    %Save all objective values and timings
    iterations.obj_vals_d = [];
    iterations.obj_vals_z = [];
    iterations.tim_vals = [];
    iterations.it_vals = [];
    
    %Save all initial vars
% %     iterations.obj_vals_d(1) = obj_val_filter;
% %     iterations.obj_vals_z(1) = obj_val_z;
% %     iterations.tim_vals(1) = 0;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Iteration for local back and forth
    max_it_d = 10;
    max_it_z = 10;

    %%%%%%%%%%%%% d specific
    Dbar = zeros(size_d_full);
    Udbar = zeros(size_d_full);    
    d_D = repmat({zeros(size_d_full)},N,1);
    %%%%%%%%%%%%%%%%%% z specific %%%%%%%
    d_Z =  zeros(size_z);    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ndim = length( size_z ) - 2;
    ss = prod(size_z(1:ndim));
    zhat_mat = repmat({ones(1,ss)},N,1);
    zhat_inv_mat = repmat({ones(1,ss)},N,1);
    
    %Iterate
    for i = 1:max_it              
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
        %obj_val_min = min(obj_val_filter, obj_val_z);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%% D
        %tic;
        for nn=1:N
            fprintf('Starting D preprocessing iterations: %d! \n', nn);
            zup{nn} = z_hat(:,:,:,:,(nn-1)*ni + 1:nn*ni) ;
            [zhat_mat{nn}, zhat_inv_mat{nn}] = precompute_H_hat_D(zup{nn}, size_z_crop, 5000); %gammas_D(2)/gammas_D(1)
        end
        %t_kernel = toc;
        fprintf('Starting D iterations after preprocessing! \n')
         
        for i_d = 1:max_it_d
            d_old = d;
            d_hat_old = d_hat;
            %tic;
            u_D2 = ProxKernelConstraint( Dbar + Udbar );
            for nn = 1:N
                %fprintf('Iter D %d, nn %d\n', i_d, nn);
                d_D{nn} = d_D{nn} + (D{nn} - u_D2);
                for ik = 1:k
                    ud_D{nn}(:,:,:,ik) = fftn( u_D2(:,:,:,ik) - d_D{nn}(:,:,:,ik)) ;
                end                
                D_hat{nn} = solve_conv_term_D(zhat_mat{nn}, zhat_inv_mat{nn}, ud_D{nn}, Bh{nn}, 5000, size_z_crop);
                for ik = 1:k
                    D{nn}(:,:,:,ik) = real(ifftn( D_hat{nn}(:,:,:,ik) ));
                end
            end
            Dbar =0; Udbar = 0;
            for nn=1:N
                Dbar = Dbar + D{nn};
                Udbar = Udbar + d_D{nn};
            end
            Dbar = (1/N)*Dbar;
            Udbar = (1/N)*Udbar;
            
%             t_kernel_tmp = toc;
%             t_kernel = t_kernel + t_kernel_tmp;
            
            d = D{1};
            d_hat = D_hat{1};
            
            d_diff = d - d_old;
            if strcmp(verbose, 'all')
                obj_val_filter = objective(z, d);
                fprintf('Iter D %d, Obj %3.3g, Diff %5.5g\n', i_d, obj_val_filter, norm(d_diff(:),2)/ norm(d(:),2));
            end
            if (norm(d_diff(:),2)/ norm(d(:),2) < tol)
                break;
            end
            
        end        
        if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
            display_func(iterate_fig, filter_fig, d, z_hat, b, size_x, size_z_crop, psf_radius, i);
        end        
        
        %%%%% Z 
%       tic;
        fprintf('Starting Z preprocessing iterations:! \n');
        [dhat_flat, dhatTdhat_flat] = precompute_H_hat_Z(d_hat, size_x);
        dhatT_flat = repmat(  conj(dhat_flat.'), [1,1,n] ); 
%       t_vars = toc;
        for i_z = 1:max_it_z
            z_old = z;
            z_hat_old = z_hat;
%           tic;
            u_Z2 = ProxSparse( z + d_Z, lambda(2) );
            d_Z = d_Z + (z - u_Z2);
            for in = 1:n
                for ik = 1:k
                    ud_Z(:,:,:,ik,in) = fftn( u_Z2(:,:,:,ik, in) - d_Z(:,:,:,ik, in)) ;
                end
            end
            z_hat = solve_conv_term_Z(dhatT_flat, dhatTdhat_flat, ud_Z, B_hat, 1, size_z);
            for in=1:n
                for ik = 1:k
                    z(:,:,:,ik, in) = real(ifftn( z_hat(:,:,:,ik, in) ));
                end
            end           
%             t_vars_tmp = toc;
%             t_vars = t_vars + t_vars_tmp;

            z_diff = z - z_old;
            if strcmp( verbose, 'all')
                obj_val_z = objective(z, d);
                fprintf('Iter Z %d, Obj %3.3g, Diff %5.5g\n', i_z, obj_val_z, norm(z_diff(:),2)/ norm(z(:),2)) %  
            end
            if (norm(z_diff(:),2)/ norm(z(:),2) < tol)
                break;
            end
        end
        if strcmp(verbose, 'brief')|| strcmp( verbose, 'all')
            display_func(iterate_fig, filter_fig, d, z_hat, b, size_x, size_z_crop, psf_radius, i);
            fprintf('Sparse coding learning loop: %d\n\n', i)
        end
        
% %         iterations.obj_vals_d(i + 1) = obj_val_filter;
% %         iterations.obj_vals_z(i + 1) = obj_val_z;
% %         iterations.tim_vals(i + 1) = iterations.tim_vals(i) + t_kernel + t_vars;
        
       %%%%% Termination %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %         if obj_val_min < obj_val_filter && obj_val_min < obj_val_z
% %             z = z_old;
% %             z_hat = z_hat_old;
% %             d = d_old;
% %             d_hat = d_hat_old;
% %             break;
% %         end
        
        if norm(z_diff(:),2)/ norm(z(:),2) < tol && norm(d_diff(:),2)/ norm(d(:),2) < tol
            break;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
    end
    
    %Final estimate
    DZ = zeros(size_x);
    for in = 1:n
        for ik = 1:k
            DZ(:,:,:,in) = DZ(:,:,:,in) + z_hat(:,:,:,ik, in)  .* d_hat(:,:,:,ik);
        end
        DZ(:,:,:,in) = real(ifftn( DZ(:,:,:,in) ));
    end  
    
    d_res = circshift(d, [psf_radius, psf_radius, psf_radius, 0] );
    d_res = d_res(1:psf_radius*2+1,1:psf_radius*2+1,1:psf_radius*2+1, :);
    z_res = z;  
    obj_val = objective(z, d);
return;

function [u_proj] = KernelConstraintProj( u, size_d, psf_radius)

    %Params
    k = size_d(end);
    ndim = length( size_d ) - 1;

    %Get support
    u_proj = circshift( u, [psf_radius, psf_radius, psf_radius, 0] ); 
    u_proj = u_proj(1:psf_radius*2+1,1:psf_radius*2+1,1:psf_radius*2+1,:);
    
     %Normalize
    for ik = 1:k
        u_curr = eval(['u_proj(' repmat(':,',1,ndim), sprintf('%d',ik), ')']);
        u_norm = sum(reshape(u_curr.^2 ,[],1));
        if u_norm >= 1
            u_curr = u_curr ./ sqrt(u_norm);
        end
        eval(['u_proj(' repmat(':,',1,ndim), sprintf('%d',ik), ') = u_curr;']);
    end
    
    %Now shift back and pad again
    u_proj = padarray( u_proj, [size_d(1:end - 1) - (2*psf_radius+1), 0], 0, 'post');
    u_proj = circshift(u_proj, -[repmat(psf_radius, 1, ndim), 0]);
    
return;

function [zhat_mat, zhat_inv_mat] = precompute_H_hat_D(z_hat, size_z, rho)
% Computes the spectra for the inversion of all H_i

%Params
ni = size_z(end);
k = size_z(end - 1);
ndim = length( size_z ) - 2;
ss = prod(size_z(1:ndim));

%Precompute spectra for H
zhat_mat = reshape( num2cell( permute( reshape(z_hat, [ss, k, ni] ), [3,2,1] ), [1 2] ), [1 ss]); %n * k * s

%Precompute the inverse matrices for each frequency
zhat_inv_mat = reshape( cellfun(@(A)(1/rho * eye(k) - 1/rho * A'*pinv(rho * eye(ni) + A * A')*A), zhat_mat, 'UniformOutput', false'), [1 ss]);

return;

function [dhat_flat, dhatTdhat_flat] = precompute_H_hat_Z(dhat, size_x )
% Computes the spectra for the inversion of all H_i

%Params
ndim = length( size_x ) - 1;
ss = prod(size_x(1:ndim));

%Precompute the dot products for each frequency
dhat_flat = reshape( dhat, ss, [] );
dhatTdhat_flat = sum(conj(dhat_flat).*dhat_flat,2);

return;

function d_hat = solve_conv_term_D(zhat_mat, zhat_inv_mat, d, B, rho, size_z )

    % Solves sum_j gamma_i/2 * || H_j d - xi_j ||_2^2
    % In our case: 1/2|| Zd - xi_1 ||_2^2 + rho * 1/2 * || d - xi_2||
    % with rho = gamma(2)/gamma(1)
    
    %Size
    n = size_z(end);
    ni=size(B,4);
    k = size_z(end - 1);
    ndim = length( size_z ) - 2;
    ss = prod(size_z(1:ndim));
    
    xi_hat_1_cell = num2cell( permute( reshape(B, ss, ni), [2,1] ), 1);
    xi_hat_2_cell = num2cell( permute( reshape(d, ss, k), [2,1] ), 1);
    
    %Invert
    x = cellfun(@(Sinv, A, b, c)(Sinv * (A' * b + rho * c)), zhat_inv_mat, zhat_mat,...
                                    xi_hat_1_cell, xi_hat_2_cell, 'UniformOutput', false);
    
    %Reshape to get back the new Dhat
    ss_size = size_z(1:ndim);
    d_hat = reshape( permute(cell2mat(x), [2,1]), [ss_size,k] );

return;

function z_hat = solve_conv_term_Z(dhatT, dhatTdhat, z, B, gammas, size_z )


    % Solves sum_j gamma_i/2 * || H_j z - xi_j ||_2^2
    % In our case: 1/2|| Dz - xi_1 ||_2^2 + rho * 1/2 * || z - xi_2||
    % with rho = gamma(2)/gamma(1)
    
    %Size
    ni = size_z(end);
    k = size_z(end - 1);
    ndim = length( size_z ) - 2;
    ss = prod(size_z(1:ndim));
    
    %Rho
    rho = gammas;
    
    %Compute b
    b = dhatT .* permute( repmat( reshape(B, ss, 1, ni), [1,k,1] ), [2,1,3] ) + rho .* permute( reshape(z, ss, k, ni), [2,1,3] );
    
    %Invert
    z_hat = 1/rho *b - 1/rho * repmat( ones([1,ss]) ./ ( rho * ones([1,ss]) + dhatTdhat.' ), [k,1,ni] ) .* dhatT .* repmat( sum(conj(dhatT).*b, 1), [k,1,1] );
    
    %Final transpose gives z_hat
    z_hat = reshape(permute(z_hat, [2,1,3]), size_z);

return;

function f_val = objectiveFunction(z, d, b, lambda_residual, lambda, psf_radius, size_z, size_x)
    
    %Params
    n = size_x(end);
    k = size_z(end-1);
    ndim = length( size_z ) - 2;
    Dz = zeros( size_x );
    all_dims = repmat(':,',1,ndim);
    
    for ik = 1:k
        eval( ['d_hat(' all_dims 'ik) = fftn( d(' all_dims 'ik) );'] );
    end
    for in = 1:n
        for ik = 1:k
            eval( ['z_hat(' all_dims 'ik, in) = fftn( z(' all_dims 'ik, in) );'] );
        end
    end
    %Dataterm and regularizer
% %     for in = 1:n
% %         Dz(:) = 0;
% %         for ik = 1:k
% %             eval( ['Dz(' all_dims(1:end-2) ':) = Dz(' all_dims(1:end-2) ':) + fftn( z(' all_dims 'ik, in) )  .* d_hat(' all_dims 'ik);'] );
% %         end
% %         eval( ['Dz(' all_dims(1:end-2) ':) = real(ifftn( Dz(' all_dims(1:end-2) ':) ));'] );
    for in = 1:n
        for ik = 1:k
            eval( ['Dz(' all_dims 'in) = Dz(' all_dims 'in) + z_hat(' all_dims 'ik, in)  .* d_hat(' all_dims 'ik);'] );
        end
        eval( ['Dz(' all_dims 'in) = real(ifftn( Dz(' all_dims 'in) ));'] );
    end  
    Dz = eval(['Dz(' repmat('1 + psf_radius:end - psf_radius,',1,ndim) ':)']);
    f_z = lambda_residual * 1/2 * norm( reshape(  Dz - eval(['b(' repmat(':,',1,ndim), ':)']), [], 1) , 2 )^2;
        
    g_z = lambda * sum( abs( z(:) ), 1 );
    
    %Function val
    f_val = f_z + g_z;
    
    
return;

function [] = display_func(iterate_fig, filter_fig, d, z_hat, b, size_x, size_z, psf_radius, iter)

    %Params
    n = size_x(end);
    k = size_z(end-1);
    ndim = length( size_z ) - 2;

    figure(iterate_fig); 
    Dz = zeros( size_x );
    all_dims = repmat(':,',1,ndim);
    
    for ik = 1:k
        eval( ['d_hat(' all_dims 'ik) = fftn( d(' all_dims 'ik) );'] );
    end
    for in = 1:n
        for ik = 1:k
            eval( ['Dz(' all_dims 'in) = Dz(' all_dims 'in) + z_hat(' all_dims 'ik, in)  .* d_hat(' all_dims 'ik);'] );
        end
        eval( ['Dz(' all_dims 'in) = real(ifftn( Dz(' all_dims 'in) ));'] );
    end  
    Dz = eval(['Dz(' repmat('1 + psf_radius:end - psf_radius,',1,ndim) ':)']);
  
    %Display some
    inds = repmat({6}, 1, ndim + 1);
    inds{1} = ':'; %Pick first two dims to show in 2D window
    inds{2} = ':';

    inds{end} = 1;
    subplot(3,2,1), imagesc(b(inds{:}));  axis image, colormap gray, title('Orig');
    subplot(3,2,2), imagesc(Dz(inds{:})); axis image, colormap gray; title(sprintf('Local iterate %d',iter));
    inds{end} = 2;
    subplot(3,2,3), imagesc(b(inds{:}));  axis image, colormap gray;
    subplot(3,2,4), imagesc(Dz(inds{:})); axis image, colormap gray;
    inds{end} = 3;
    subplot(3,2,5), imagesc(b(inds{:}));  axis image, colormap gray;
    subplot(3,2,6), imagesc(Dz(inds{:})); axis image, colormap gray;
    
    figure(filter_fig);
    sqr_k = ceil(sqrt(k));
    pd = 1;
    d_disp = zeros( sqr_k * [psf_radius*2+1 + pd, psf_radius*2+1 + pd] + [pd, pd]);
    inds = repmat({10}, 1, ndim + 1);
    inds{1} = ':'; %Pick first two dims to show in 2D window
    inds{2} = ':';
    for j = 0:k - 1
        inds{end} = j + 1;
        d_curr = circshift( d(inds{:}), [psf_radius, psf_radius] ); 
        d_curr = d_curr(1:psf_radius*2+1, 1:psf_radius*2+1);
        d_disp( floor(j/sqr_k) * (size(d_curr,1) + pd) + pd + (1:size(d_curr,1)) , mod(j,sqr_k) * (size(d_curr,2) + pd) + pd + (1:size(d_curr,2)) ) = d_curr;
    end
    imagesc(d_disp), colormap gray; axis image; colorbar; title(sprintf('Local filter iterate %d',iter));
    drawnow;        
return;
