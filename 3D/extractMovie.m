%Clear
clear;
close all;

%Define filenames
fn = './data_videos/drift_hungary_1280x720.mp4';
start_frame = 4090;
nFrames = 140;

%fn = './data_videos/legend_4k-_iceland_in_ultra_hd_1920x1080.mp4';
%start_frame = 5400; %6000;
%nFrames = 100;

%fn = './data_videos/navegar_la_reina_morsa_1280x720_2.mp4';
%start_frame = 1450;
%nFrames = 200;

%Verbose
verbose = 'brief';
debug = false;

%Create sampling parameters
dimmings = 1;

%Prepare output folder
fprintf('\n\n#### Processing frames [%d/%d] of %s ####\n\n', start_frame, start_frame + nFrames, fn)
[input_path, input_filename, input_ext] = fileparts( fn );

outdir = sprintf('./movs/%s', input_filename);
mkdir( outdir );

%Start opening the readers and writers
movObj = VideoReader(fn);

vidFrames = movObj.NumberOfFrames;
vidHeight = movObj.Height;
vidWidth = movObj.Width;
vidFrameRate = movObj.FrameRate;
mov_Frame = struct('cdata', zeros(vidHeight, vidWidth, 3, 'uint8'), 'colormap', []);

%Movie res
height = 300;
mov_data = zeros(height, round( vidWidth * (height/vidHeight) ), 3, nFrames);

%Debug show movie
show_movie = true;
if  show_movie
    
    %Read full movie first
    for k = 1 : nFrames
        
        frame = im2double( read(movObj, start_frame + k) );
        frame = imresize( frame , [height, round(  vidWidth * (height/vidHeight) ) ], 'bicubic' );
        frame = max(frame, 0);
        mov_data(:,:,:,k) = frame;
        
    end

    figure();
    for t = 1:nFrames
       imshow( mov_data(:,:,:,t) ), title(sprintf('Frame [%d/%d]', t, nFrames ));
       pause(1/30);
    end
end

%Select a little window for a single pixel test
px = 200;
py = 200;
wnd_r = 10;
%mov_data = mov_data( py - wnd_r:py + wnd_r, px - wnd_r:px + wnd_r, :, : );

%Show
figure();
hold on;
plot( reshape( mov_data(wnd_r + 1, wnd_r + 1, 1, :), [], 1), '-r' );
plot( reshape( mov_data(wnd_r + 1, wnd_r + 1, 2, :), [], 1), '-g' );
plot( reshape( mov_data(wnd_r + 1, wnd_r + 1, 3, :), [], 1), '-b' );
hold off;
xlabel('Frames'), ylabel('Intensity');
legend('Red', 'Green', 'Blue');

%Save the results
save( sprintf('%s/full_movie.mat', outdir) , 'mov_data' );
fprintf('\n\nOverall movie reconstruction done, result saved.\n\n')
