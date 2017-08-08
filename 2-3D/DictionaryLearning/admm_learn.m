function [ d_res, z_res, Dz, obj_val ] = admm_learn(b, kernel_size,...
                    lambda_residual, lambda_prior, ...
                    max_it, tol, ...
                    verbose, init, smooth_init)

    %Kernel size contains kernel_size = [psf_s, psf_s, k]
    psf_s = kernel_size(1);
    k = kernel_size(end);
    n = size(b,5);
                
    %PSF estimation
    psf_radius = floor( psf_s/2 );
    size_x = [size(b,1) + 2*psf_radius, size(b,2) + 2*psf_radius, size(b,3), size(b,4), n];
    size_z = [size_x(1), size_x(2), k, n];
    size_k_full = [size_x(1), size_x(2), size_x(3), size_x(4), k]; 
    size_zhat = [size_x(1), size_x(2), 1, 1, k, n];
       
    %Smooth offset
    smoothinit = padarray( smooth_init, [psf_radius, psf_radius, 0, 0, 0], 'symmetric', 'both');
    
    % Objective
    objective = @(z, dh) objectiveFunction( z, dh, b, lambda_residual, lambda_prior, psf_radius, size_z, size_x, smoothinit );
 
    %Prox for masked data
    [M, Mtb] = precompute_MProx(b, psf_radius, smoothinit); %M is MtM
    ProxDataMasked = @(u, theta) (Mtb + 1/theta * u ) ./ ( M + 1/theta * ones(size_x) ); 
    
    %Prox for sparsity
    ProxSparse = @(u, theta) max( 0, 1 - theta./ abs(u) ) .* u;
    
    %Prox for kernel constraints
    ProxKernelConstraint = @(u) KernelConstraintProj( u, size_k_full, psf_radius);
    
    %% Pack lambdas and find algorithm params
    lambda = [lambda_residual, lambda_prior];
    gamma_heuristic = 60 * lambda_prior * 1/max(b(:));
    gammas_D = [gamma_heuristic / 5000, gamma_heuristic]; 
    gammas_Z = [gamma_heuristic / 500, gamma_heuristic]; 
    
    %% Initialize variables for K
    varsize_D = {size_x, size_k_full};
    xi_D = { zeros(varsize_D{1}), zeros(varsize_D{2}) };
    xi_D_hat = { zeros(varsize_D{1}), zeros(varsize_D{2}) };
    
    u_D = { zeros(varsize_D{1}), zeros(varsize_D{2}) };
    d_D = { zeros(varsize_D{1}), zeros(varsize_D{2}) };
    v_D = { zeros(varsize_D{1}), zeros(varsize_D{2}) };

    %Initial iterates
    if ~isempty(init)
        d_hat = init.d;
        d = [];
    else
        d = padarray( randn(kernel_size([1 2 5])), [size_x(1) - kernel_size(1), size_x(2) - kernel_size(2), 0], 0, 'post');
        d = circshift(d, -[psf_radius, psf_radius, 0] );
        d = permute(repmat(d, [1 1 1 kernel_size(3) kernel_size(4)]), [1 2 4 5 3]);
        d_hat = fft2(d);
    end
    
    %% Initialize variables for Z
    varsize_Z = {size_x([1 2 3 4 5]), size_z};
    xi_Z = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };
    xi_Z_hat = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };
    
    u_Z = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };
    d_Z = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };
    v_Z = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };
    
    z = randn(size_z);
    
    %Initial vals
    obj_val = objective(z, d_hat);
    
    if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
        fprintf('Iter %d, Obj %3.3g, Diff %5.5g\n', 0, obj_val, 0)
    end
    
    %Iteration for local back and forth
    max_it_d = 10;
    max_it_z = 10;
    
    obj_val_filter = obj_val;
    obj_val_z = obj_val;
    
    %Iterate
    for i = 1:max_it
        %% Update kernels
        %Timing
        tic;
        
        %Recompute what is necessary for kernel convterm later
        rho = gammas_D(2)/gammas_D(1);
        obj_val_min = min(obj_val_filter, obj_val_z);
        d_old = d;
        d_hat_old = d_hat;

        %Timing
        t_kernel = toc;
        z_hat = fft2(z); %reshape( fft2(reshape(z, size_z(1),size_z(2),size_z(3),[])), size_zhat );

        for i_d = 1:max_it_d

            %Timing
            tic;
            
            %Compute v_i = H_i * z
            v_D{1} = real(ifft2( reshape(sum( bsxfun(@times, d_hat, permute(z_hat, [1 2 5 6 3 4])), 5), size_x) ));
            v_D{2} = d;

            %Compute proximal updates
            u_D{1} = ProxDataMasked( v_D{1} - d_D{1}, lambda(1)/gammas_D(1) );
            u_D{2} = ProxKernelConstraint( v_D{2} - d_D{2});

            for c = 1:2
                %Update running errors
                d_D{c} = d_D{c} - (v_D{c} - u_D{c});

                %Compute new xi and transform to fft
                xi_D{c} = u_D{c} + d_D{c};
                xi_D_hat{c} = fft2(xi_D{c});
            end

            %Solve convolutional inverse
            d_hat = solve_conv_term_D(z_hat, xi_D_hat, rho, size_z);
            d = real(ifft2( d_hat ));
            
            %Timing
            t_kernel_tmp = toc;
            t_kernel = t_kernel + t_kernel_tmp;
            
            obj_val = objective(z, d_hat);
            if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
                fprintf('--> Obj %5.5g \n', obj_val )
            end
        end
        
%         obj_val_filter_old = obj_val_old;
        obj_val_filter = obj_val;

        %Debug progress
        d_diff = d - d_old;
        d_comp = d;
        if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
            obj_val = objective(z, d_hat);
            fprintf('Iter D %d, Obj %5.5g, Diff %5.5g\n', i, obj_val, norm(d_diff(:),2)/ norm(d_comp(:),2))
        end
        
        %% Update sparsity term
        
        %Timing
        tic;

        %Recompute what is necessary for convterm later
        [dhat_flat, dhatTdhat_flat] = precompute_H_hat_Z(d_hat, size_x);
        dhatT_flat = conj(permute(dhat_flat, [3 2 1])); %Same for all images
        
        z_hat = fft2(z);
        z_old = z;
        z_hat_old = z_hat;

        %Timing
        t_vars = toc;

        for i_z = 1:max_it_z
            
            %Timing
            tic;

            %Compute v_i = H_i * z
            v_Z{1} = real(ifft2(squeeze(sum(bsxfun(@times, d_hat, permute(z_hat, [1 2 5 6 3 4])), 5))));
            v_Z{2} = z;

            %Compute proximal updates
            u_Z{1} = ProxDataMasked( v_Z{1} - d_Z{1}, lambda(1)/gammas_Z(1) );
            u_Z{2} = ProxSparse( v_Z{2} - d_Z{2}, lambda(2)/gammas_Z(2) );

            for c = 1:2
                %Update running errors
                d_Z{c} = d_Z{c} - (v_Z{c} - u_Z{c});

                %Compute new xi and transform to fft
                xi_Z{c} = u_Z{c} + d_Z{c};
                xi_Z_hat{c} = fft2( xi_Z{c} );
            end

            %Solve convolutional inverse
            z_hat = solve_conv_term_Z(dhatT_flat, dhatTdhat_flat, xi_Z_hat, gammas_Z, size_z, kernel_size(3)*kernel_size(4));
            z = real(ifft2(z_hat));

            %Timing
            t_vars_tmp = toc;
            t_vars = t_vars + t_vars_tmp;
            
            obj_val = objective(z, d_hat);
            if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
                fprintf('--> Obj %5.5g \n', obj_val )
            end
        
        end
        
        obj_val_z = obj_val;
        
%         if obj_val_min <= obj_val_filter && obj_val_min <= obj_val_z
%             z_hat = z_hat_old;
%             z = reshape( real(ifft2( reshape(z_hat, size_x(1), size_x(2), []) )), size_z );
%             
%             d_hat = d_hat_old;
%             d = real(ifft2( d_hat ));
%             
%             obj_val = objective(z, d_hat);
%             break;
%         end
        
        %Debug progress
        z_diff = z - z_old;
        z_comp = z;
        if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
            fprintf('Iter Z %d, Obj %5.5g, Diff %5.5g, Sparsity %5.5g\n', i, obj_val, norm(z_diff(:),2)/ norm(z_comp(:),2), nnz(z(:))/numel(z(:)))
        end
        
        %Termination
        if norm(z_diff(:),2)/ norm(z_comp(:),2) < tol && norm(d_diff(:),2)/ norm(d_comp(:),2) < tol
            break;
        end
    end
    
    %Final estimate
    z_res = z;
    
    d_res = circshift( d, [psf_radius, psf_radius, 0, 0] ); 
    d_res = d_res(1:psf_radius*2+1, 1:psf_radius*2+1, :, :);
    
    z_hat = reshape(fft2(z), size_zhat);
    Dz = real(ifft2( reshape(sum(bsxfun(@times, d_hat, z_hat), 5), size_x) )) + smoothinit;
    
return;

function [u_proj] = KernelConstraintProj( u, size_k_full, psf_radius)
    
    %Get support
    u_proj = circshift( u, [psf_radius, psf_radius, 0, 0] ); 
    u_proj = u_proj(1:psf_radius*2+1, 1:psf_radius*2+1, :, :, :);
    
    %Normalize
    u_norm = repmat( sum(sum(u_proj.^2, 1),2), [size(u_proj,1), size(u_proj,2), 1, 1] );
    u_proj( u_norm >= 1 ) = u_proj( u_norm >= 1 ) ./ sqrt(u_norm( u_norm >= 1 ));
    
    %Now shift back and pad again
    u_proj = padarray( u_proj, (size_k_full - size(u_proj)), 0, 'post');
    u_proj = circshift(u_proj, -[psf_radius, psf_radius, 0, 0] );

return;

function [M, Mtb] = precompute_MProx(b, psf_radius,smoothinit)
    
    M = padarray(ones(size(b)), [psf_radius, psf_radius, 0, 0]);
    Mtb = padarray(b, [psf_radius, psf_radius, 0, 0]).*M - smoothinit.*M;
    
return;

function [dhat_flat, dhatTdhat_flat] = precompute_H_hat_Z(dhat, size_x )
% Computes the spectra for the inversion of all H_i

%Precompute the dot products for each frequency
dhat_flat = reshape( dhat, size_x(1) * size_x(2), size_x(3)*size_x(4), [] );
dhatTdhat_flat = sum(conj(dhat_flat).*dhat_flat,3);

return;

%Rho
%rho = gammas(2)/gammas(1);
function d_hat = solve_conv_term_D(z_hat, xi_hat, rho, size_z )

    % Solves sum_j gamma_i/2 * || H_j d - xi_j ||_2^2
    % In our case: 1/2|| Zd - xi_1 ||_2^2 + rho * 1/2 * || d - xi_2||
    % with rho = gamma(2)/gamma(1)
    
    %Size
    sy = size_z(1); sx = size_z(2); sw = size(xi_hat{1},3)*size(xi_hat{1},4); k = size_z(3); n = size_z(4);

    %Reshape to cell per frequency
    xi_hat_1_cell = num2cell( permute( reshape(xi_hat{1}, sx * sy * sw, n), [2,1] ), 1);
    xi_hat_2_cell = num2cell( permute( reshape(xi_hat{2}, sx * sy * sw, k), [2,1] ), 1);
    zhat_mat = reshape( num2cell( permute( reshape(z_hat, [sy*sx, k, n] ), [3,2,1] ), [1 2] ), [1 sy*sx]); %n * k * s

    %Invert
    x = cell(size(xi_hat_1_cell));
    for i=1:sx*sy
        opt = (1/rho * eye(k) - 1/rho * zhat_mat{i}'*pinv(rho * eye(n) + zhat_mat{i}*zhat_mat{i}')*zhat_mat{i});
        for j=1:sw
            ind = (j-1)*sx*sy + i;
            x{ind} = opt*(zhat_mat{i}' * xi_hat_1_cell{ind} + rho * xi_hat_2_cell{ind});
        end
    end

    %Reshape to get back the new Dhat
    d_hat = reshape( permute(cell2mat(x), [2,1]), size(xi_hat{2}) );

return;

function z_hat = solve_conv_term_Z(dhatT, dhatTdhat, xi_hat, gammas, size_z, sw )


    % Solves sum_j gamma_i/2 * || H_j z - xi_j ||_2^2
    % In our case: 1/2|| Dz - xi_1 ||_2^2 + rho * 1/2 * || z - xi_2||
    % with rho = gamma(2)/gamma(1)
    sy = size_z(1); sx = size_z(2); k = size_z(3); n = size_z(4);
    
    %Rho
    rho = sw * gammas(2)/gammas(1);
    
    %Compute b
    b = squeeze(sum(bsxfun(@times, dhatT, permute(reshape(xi_hat{1}, sy*sx, sw, n), [4,2,1,3])),2)) + rho .* permute( reshape(xi_hat{2}, sy*sx, k, n), [2,1,3] );

    %Invert
    scInverse = ones([1,sx*sy]) ./ ( rho * ones([1,sx*sy]) + sum(dhatTdhat,2).');

    x = 1/rho*b - 1/rho * bsxfun(@times, bsxfun(@times, scInverse, squeeze(sum(dhatTdhat,2))'), b);

    %Final transpose gives z_hat
    z_hat = reshape(permute(x, [2,1,3]), size_z);

return;

function f_val = objectiveFunction( z, d_hat, b, lambda_residual, lambda, psf_radius, size_z, size_x, smoothinit)
    
    %Params
%     n = size_z(4);

    %Data term and regularizer
%     z2 = permute(repmat(z, [1 1 1 1 size_x(3)]), [1 2 5 3 4]);
%     zhat = reshape( fft2(reshape(z2,size_z(1),size_z(2),size_z(3),[])), size(z2) );
%     Dz = real(ifft2( reshape(sum(repmat(d_hat,[1,1,1,1,n]) .* zhat, 4), size_x) )) + smoothinit;
    
    z_hat = permute(fft2(z), [1 2 5 6 3 4]);
    Dz = real(ifft2( squeeze(sum(bsxfun(@times, d_hat, z_hat),5)) )) + smoothinit;
    
    f_z = lambda_residual * 1/2 * norm( reshape( Dz(1 + psf_radius:end - psf_radius, ...
            1 + psf_radius:end - psf_radius,:,:,:) - b, [], 1) , 2 )^2;
    g_z = lambda * sum( abs( z(:) ), 1 );
    
    %Function val
    f_val = f_z + g_z;
    
return;
