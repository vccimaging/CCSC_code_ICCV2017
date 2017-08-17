function [z, res] = admm_solve_video_weighted_sampling(b, kmat , mask, ...
                    lambda_residual, lambda_prior, max_it, tol, verbose, psf, smooth_init)
                   
                
    k_dirac = zeros(size(kmat,1), size(kmat,2), size(kmat,3));
    k_dirac( floor(size(kmat,1)/2) + 1, floor(size(kmat,2)/2) + 1, floor(size(kmat,3)/2) + 1 ) = 1; 
    kmat = cat(4, k_dirac, kmat );
                
    %Precompute spectra for H (assumed to be convolutional)
    psf_radius = floor( [size(kmat,1)/2, size(kmat,2)/2, size(kmat,3)/2] );
    size_x = [size(b,1) + 2*psf_radius(1), size(b,2) + 2*psf_radius(2), size(b,3) + 2*psf_radius(3)];
    [dhat_k, dhat, dhat_flat, dhatTdhat_flat] = precompute_H_hat(kmat, psf, size_x);
    dhatT_flat = conj(permute(dhat_flat, [2 1]));
    
    %Smooth offset
    smoothinit = padarray( smooth_init, psf_radius, 'symmetric', 'both');

    %Size of z is now the padded array
    size_z = [size_x(1), size_x(2), size_x(3), size(kmat, 4)];
    
    % Objective
    objective = @(v) objectiveFunction( v, dhat, b, mask, lambda_residual, lambda_prior, psf_radius, smoothinit );
    
    %Proximal terms
    conv_term = @(xi_hat, gammas) solve_conv_term(dhatT_flat, dhatTdhat_flat, xi_hat, gammas, size_z);
    
    %Prox for masked data
    [M, Mtb] = precompute_MProx(b, mask, psf_radius, smoothinit); %M is MtM
    ProxDataMasked = @(u, theta) (Mtb + 1/theta * u ) ./ ( M + 1/theta * ones(size_x) ); 
    
    %Prox for sparsity
    ProxSparse = @(u, theta) max( 0, 1 - theta./ abs(u) ) .* u;
    
    %Pack lambdas and find algorithm params
    lambda = [lambda_residual, lambda_prior];
    gamma_heuristic = 500 * lambda_prior * 1/max(b(:));
    gamma = [gamma_heuristic, gamma_heuristic];
    
    %Initialize variables
    varsize = {size_x, size_z};
    xi = { zeros(varsize{1}), zeros(varsize{2}) };
    xi_hat = { zeros(varsize{1}), zeros(varsize{2}) };
    
    u = { zeros(varsize{1}), zeros(varsize{2}) };
    d = { zeros(varsize{1}), zeros(varsize{2}) };
    v = { zeros(varsize{1}), zeros(varsize{2}) };
    
    %Initial iterate
    z = zeros(varsize{2});
    z_hat = zeros(varsize{2});
    
    if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
        obj_val = objective(z);
        fprintf('Iter %d, Obj %3.3g, Diff %5.5g\n', 0, obj_val, 0)
    end
    
    %Iterate
    for i = 1:max_it
        
        %Compute v_i = H_i * z
        v{1} = real(ifftn( sum( dhat .* z_hat, 4)));
        v{2} = z;
        
        %Compute proximal updates
        u{1} = ProxDataMasked( v{1} - d{1}, lambda(1)/gamma(1) );
        u{2} = ProxSparse( v{2} - d{2}, lambda(2)/gamma(2) );

        for c = 1
            %Update running errors
            d{c} = d{c} - (v{c} - u{c});

            %Compute new xi and transform to fft
            xi{c} = u{c} + d{c};
            xi_hat{c} = fftn(xi{c});
        end
        for c = 2
            %Update running errors
            d{c} = d{c} - (v{c} - u{c});

            %Compute new xi and transform to fft
            xi{c} = u{c} + d{c};
            
            for q=1:size(z,4)
                xi_hat{c}(:,:,:,q) = fftn(xi{c}(:,:,:,q));
            end
            
%             xi_hat{c} = fftn(xi{c});
        end

        %Solve convolutional inverse
        zold = z;
        z_hat = conv_term( xi_hat, gamma );
        for q=1:size(z,4)
            z(:,:,:,q) = real(ifftn(z_hat(:,:,:,q)));
        end
%         z = real(ifftn( z_hat ));
       
        z_diff = z - zold;
        z_comp = z;
        if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
            obj_val = objective(z);
            fprintf('Iter %d, Obj %3.3g, Diff %5.5g\n', i, obj_val, norm(z_diff(:),2)/ norm(z_comp(:),2))
        end
        if norm(z_diff(:),2)/ norm(z_comp(:),2) < tol
            break;
        end
    end
    
    Dz = real(ifftn( sum( bsxfun(@times, dhat_k, z_hat), 4))) + smoothinit;
    res = Dz(1 + psf_radius(1):end - psf_radius(1),1 + psf_radius(2):end - psf_radius(2),1 + psf_radius(3):end - psf_radius(3));
    
return;

function [M, Mtb] = precompute_MProx(b, mask, psf_radius, smoothinit)
    
    M = padarray(mask, psf_radius, 0, 'both');
    Mtb = padarray(b, psf_radius, 0, 'both') .* M - smoothinit .* M;
    
return;

function [dhat_k, dhat, dhat_flat, dhatTdhat_flat] = precompute_H_hat(kmat, psf, size_x )
% Computes the spectra for the inversion of all H_i

psf_hat = psf2otf(psf, size_x);

%Precompute spectra for H
dhat = zeros( [size_x(1), size_x(2), size_x(3), size(kmat,4)] );
dhat_k = zeros( [size_x(1), size_x(2), size_x(3), size(kmat,4)] );
for i = 1:size(kmat,4)  
    dhat_k(:,:,:,i)  = psf2otf(kmat(:,:,:,i), [size_x(1) size_x(2) size_x(3)]);
    dhat(:,:,:,i) = psf_hat .* psf2otf(kmat(:,:,:,i), [size_x(1) size_x(2) size_x(3)]);
end

%Precompute the dot products for each frequency
dhat_flat = reshape( dhat, size_x(1) * size_x(2) * size_x(3), [] );
dhatTdhat_flat = sum(sum(conj(dhat_flat).*dhat_flat,3),2);

return;

function z_hat = solve_conv_term(dhatT, dhatTdhat, xi_hat, gammas, size_z)


    % Solves sum_j gamma_i/2 * || H_j z - xi_j ||_2^2
    % In our case: 1/2|| Dz - xi_1 ||_2^2 + rho * 1/2 * || z - xi_2||
    % with rho = gamma(2)/gamma(1)
    sy = size_z(1); sx = size_z(2); k = size_z(4); sw = size(xi_hat{1},3);
    
    %Rho
    rho = sw * gammas(2)/gammas(1);
    
    %Compute b
    b = bsxfun(@times, dhatT, reshape(xi_hat{1}, 1, sy*sx*sw)) + rho .* reshape(xi_hat{2}, sy*sx*sw, k)';

    %Invert
    scInverse = ones([1,sx*sy*sw]) ./ ( rho * ones([1,sx*sy*sw]) + dhatTdhat.');
    x = 1/rho*b - 1/rho * bsxfun(@times, bsxfun(@times, scInverse, dhatTdhat.'), b);

    %Final transpose gives z_hat
    z_hat = reshape(permute(x, [2,1,3]), size_z);

return;

function f_val = objectiveFunction( z, dhat, b, mask, lambda_residual, lambda, psf_radius, smoothinit)
    
    %Dataterm and regularizer
    zh = zeros(size(z));
    for i=1:size(z,4)
        zh(:,:,:,i) = fftn(z(:,:,:,i));
    end
    
    Dz = real(ifftn( sum( bsxfun(@times, dhat, zh), 4))) + smoothinit;
    f_z = lambda_residual * 1/2 * norm( reshape( mask .* Dz(1 + psf_radius(1):end - psf_radius(1),1 + psf_radius(2):end - psf_radius(2),1 + psf_radius(3):end - psf_radius(3)) - mask .* b, [], 1) , 2 )^2;
    g_z = lambda * sum( abs( z(:) ), 1 );
    
    %Function val
    f_val = f_z + g_z;
    
return;


