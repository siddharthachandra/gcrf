VOCPATH = '/data2/datasets/VOC2012';
list = textread('resources/filelists/test.txt','%s');
n = numel(list);
for i = 1: n
    fprintf('Flipping image %d/%d\n',i,n);
    imgpath = fullfile(VOCPATH,list{i});
    img = imread(imgpath);
    flip = img(:,end:-1:1,:);
    imwrite(flip,regexprep(imgpath,'.jpg','_f.jpg'));
end

