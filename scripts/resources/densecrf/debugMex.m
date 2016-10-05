make all debug;
addpath('build'); addpath('examples');
im = imread('examples/im1.ppm'); anno = imread('examples/anno1.ppm');
opts.imHeight = size(im,1);
opts.imWidth  = size(im,2);
unary = zeros(21, opts.imHeight*opts.imWidth,'single');
[opts.nLabels] = size(unary,1);
% dbmex on;
res = dense_inference_mex(unary,vec(permute(im,[3,2,1])),vec(permute(anno, [3 2 1])),opts);
res = permute(reshape(res, 21, opts.imWidth, opts.imHeight), [3 2 1]);

% res = dense_inference_mex_old(unary,im(:),anno(:),opts);
% res = permute(reshape(res, 21, opts.imWidth, opts.imHeight), [3 2 1]);
