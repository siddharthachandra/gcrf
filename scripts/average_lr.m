SCORESDIR = 'results/release';
imglist   = 'resources/filelists/test_id_both.txt';
IMGDIR    = '/data2/datasets/VOC2012/JPEGImages';
OUTDIR    = 'results/scores';
IMOUTDIR  = 'results/nocrf';
imgs      = textread(imglist,'%s');
n         = 1456;
load resources/pascal_seg_colormap.mat
for i = 1 : n
    fprintf('Doing %d/%d\n',i,n);
    impath= fullfile(IMGDIR, [imgs{i} '.jpg']);
    impatf= fullfile(IMGDIR, [imgs{i} '_f.jpg']);
    im    = imread(impath);
    score = load(fullfile(SCORESDIR,[imgs{i} '_blob_0.mat']));
    scoref= load(fullfile(SCORESDIR,[imgs{i} '_f_blob_0.mat']));
    [r_,c_,d_] = size(im);
    score = permute(score.data,[2 1 3]);
    scoref= permute(scoref.data,[2 1 3]);
    score = score(1:r_,1:c_,:);
    scoref= scoref(1:r_,1:c_,:);
    data  = (score+scoref(:,end:-1:1,:))/2;
    save(fullfile(OUTDIR,[imgs{i} '_blob_0.mat']),'data');
    [~,pos]= max(data,[],3);
    pos   = uint8(pos-1);
    imwrite(pos,colormap,fullfile(IMOUTDIR,[imgs{i} '.png']));
end
