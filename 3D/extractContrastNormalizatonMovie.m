%Load movie
mov_file = cell(0);
mov_file{1} = './movs/drift_hungary_1280x720/full_movie.mat';
%mov_file{2} = './movs/sintel-2048-surround/full_movie.mat';
%mov_file{3} = './movs/navegar_la_reina_morsa_1280x720_2/full_movie.mat';
%mov_file{4} = './movs/legend_4k-_iceland_in_ultra_hd_1920x1080/full_movie.mat';

verbose = 'all';

%Iterate
for m = 1:length(mov_file)

    %Debug
    fprintf('Processing [%d/%d]\n', m, length(mov_file) )
    
    mov_filename = mov_file{m};
    [pathstr, name, ext] = fileparts(mov_filename);
    output_file = sprintf('%s/%s_localCN.mat', pathstr, name);
    
    I = load(mov_filename);
    I = I.mov_data;

    %Convert to gray
    I_gray = zeros( size(I,1), size(I,2), size(I,4), 'single' );
    for f = 1:size(I,4)
        I_gray(:,:,f) = single(rgb2gray(I(:,:,:,f)));
    end
    I = I_gray;

    I = local_cn(I); %% Apply contrast normalisation to video frames 
    
    %% Show the movie
    if strcmp(verbose, 'all') 
      figure();
      for f = 1:size(I,3)
        imshow(I(:,:,f),[]);
        title(sprintf('Movie [%d] Frame %03d',m,f));
        pause(0.01);
      end
    end
    
    %Now save
    save(output_file, 'I');
end
