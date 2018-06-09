start = tic;
addpath(fullfile(pwd,'lib'));

imgList = dir('ThisComp/*/cv_norm/*.png');
logfile = fopen('log','a+');

for i=1:size(imgList)
    set = tic;
    imgFile = join([imgList(i).folder,'/',imgList(i).name]);
    outFile = join([imgFile,'.mat']);
    
    fprintf(logfile, imgFile);
    fprintf(logfile, '\n');
    fprintf(logfile, 'start \n%d, month %d, %d, %d:%d:%f\n',clock);
    
    gPb_orient = globalPb_pieces(imgFile, outFile);
    
    fprintf(logfile, 'finish \n%d, month %d, %d, %d:%d:%f\n\n',clock);
end

finish = toc(start);