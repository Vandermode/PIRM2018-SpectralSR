datadir = '.'
addpath(genpath(datadir))
arch = 'emsrx3'
prefix = 'm1s1_final'
t_prefix = 'm1s1_final'

for i = 221:240  %201:220%101:110

    sr_path = fullfile(datadir, 'result', arch, prefix, ['img_',num2str(i), '.mat']);

    lr3_path = fullfile('testing_lr', ['image_', num2str(i), '_lr3.fla']);

    Data = FLAread(lr3_path);

    load(sr_path)

    Data.I = sr;

    Data.HDR.samples = 480;

    Data.HDR.lines = 240;

    mkdir(fullfile(datadir, 'envi', arch, t_prefix))

    FLAwrite(fullfile(datadir, 'envi', arch, t_prefix, ['image_', num2str(i), '_tr1.fla']), Data)

end
