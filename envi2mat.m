datadir = 'validation_lr'
addpath(genpath(datadir))

for i = 201:220%221:240

    hr_path = fullfile('validation_hr', ['image_', num2str(i), '_tr1.fla']);

    hr = FLAread(hr_path);

    hr = hr.I;

    lr2_path = fullfile(datadir, ['image_', num2str(i), '_lr2.fla']);

    lr2 = FLAread(lr2_path);

    lr2 = lr2.I;

    lr3_path = fullfile(datadir, ['image_', num2str(i), '_lr3.fla']);

    lr3 = FLAread(lr3_path);

    lr3 = lr3.I;

    outpath = fullfile('validation',['img_', num2str(i),'.mat']);

    save(outpath, 'lr2', 'lr3', 'hr');

end
