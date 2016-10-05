addpath resources/densecrf/
addpath resources/densecrf/build
a = getenv('LD_LIBRARY_PATH');
setenv('LD_LIBRARY_PATH',[a ':/usr/local/lib:/lib64']);
load('resources/pascal_seg_colormap.mat');
SCORESDIR = 'results/scores';
OUTDIR = 'results/crf';
DATASETDIR = '/data2/datasets/VOC2012';
[rgb]= textread('resources/filelists/test.txt','%s');

opts.nIter      = 10;
opts.nLabels    = 21;
opts.xstdUnary  = 1;
opts.ystdUnary  = 1;
opts.wUnary     = 3;
opts.logprob    = true;
opts.xstdBinary = 31;
opts.ystdBinary = 31;
opts.wBinary = 3;
opts.rgbStd = 4;
opts.imHeight   = 242;
opts.imWidth    = 322;
for imgID = 1 : length(rgb)
	[path_,fn_,ext_] = fileparts(rgb{imgID});
	I = imread(fullfile(DATASETDIR,rgb{imgID}));
	scores = load(fullfile(SCORESDIR,[fn_ '_blob_0.mat']));
	scores = scores.data;
	img_row = size(I, 1);
	img_col = size(I, 2);
	opts.imHeight = img_row;
	opts.imWidth = img_col;
	scores = scores(1:img_row, 1:img_col, :);
	crf = denseCRF(I,scores,10,opts);
	[~,lblsafter] = max(crf,[],3);
	lblsafter = lblsafter - 1;
	lblsafter = uint8(lblsafter);	
	imwrite(lblsafter,colormap,fullfile(OUTDIR,[fn_ '.png']));
	fprintf('Done %d/%d\n',imgID,length(rgb));
end
