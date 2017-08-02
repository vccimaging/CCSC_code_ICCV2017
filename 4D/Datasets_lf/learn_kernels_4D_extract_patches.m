% Learning test for sparse convolutional coding

clear;
close all;

%% Debug options
verbose = 'all';

%% Load the movie 
    
%Load movie
fn = '../Datasets_lf/food_localCN_bis.mat';
I = load(fn);
I = I.I;
    
n_vid = 8 * 8;
s_data = [50, 50, 5, 5];

%Check for dims
if any( s_data > size(I) )
    error('Selection too large')
end

%% Show the full movie
if strcmp(verbose, 'all') 
  figure();
  for k2 = 1:size(I,4)
      for k1 = 1:size(I,3)
          %for j = 1:size(I,3)
            imshow(I(:,:,k1, k2),[]);
            title(sprintf('Multispec [ VIEW (%d, %d) ]',k1, k2));
            pause(0.01);
          %end
      end
  end
end

%Size
size_I = size(I);

%Extract the individual movies
ndims = length(size_I);
offsets_movs = ceil( rand(ndims, n_vid) .* repmat( size_I(:) - (s_data(:) + 1), 1, n_vid) );

%Sample
b = zeros( s_data(1), s_data(2), s_data(3), s_data(4), n_vid, 'single');
for i = 1:n_vid
    offset_start = offsets_movs(:,i);
    off_start = [offset_start'; s_data];
    range_str = sprintf( repmat('%d + (1:%d),',1,ndims), off_start(:) );
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

    %Iterate over spec
    figure();     
    for k2 = 1:size(I,4)
      for k1 = 1:size(I,3)
          %for j = 1:size(I,3)
            imshow(b_view(:,:,k1, k2),[]);
            title(sprintf('Extracted Multispec [ VIEW (%d, %d) ]',k1, k2));
            pause(0.01);
          %end
      end
    end
    
    %Iterate over views
    figure();     
    %for j = 1:size(I,3)     
       for k2 = 1:size(I,4)
         for k1 = 1:size(I,3)
            imshow(b_view(:,:,k1, k2),[]);
            title(sprintf('Extracted Multispec [ VIEW (%d, %d)]',k1, k2));
            pause(0.01);
          end
      end
    %end
    
end

[pathstr, name, ext] = fileparts(fn);
output_file = sprintf('%s/%s_bis3_8x8.mat', pathstr, name);
save(output_file, 'b');
