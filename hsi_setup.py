import torch
import torch.optim as optim
import models

import os
import argparse

from os.path import join
from utility import *


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def train_options(parser):
    def _parse_str_args(args):
        str_args = args.split(',')
        parsed_args = []
        for str_arg in str_args:
            arg = int(str_arg)
            if arg >= 0:
                parsed_args.append(arg)
        return parsed_args

    parser.add_argument('--prefix', '-p', type=str,
                        default='SR', help='the prefix of savepath')    
    parser.add_argument('--arch', '-a', metavar='ARCH', required=True,
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names))
    parser.add_argument('--sf', type=int, default=2, choices=[2,3],
                            help='scale factor. Default=2')
    parser.add_argument('--patchsize', '-ps', type=int, default=48, choices=[48,64],
                            help='patch size of lr for training. Default=48')
    parser.add_argument('--batchSize', '-b', type=int,
                        default=16, help='training batch size. Default=16')
    parser.add_argument('--nEpochs', '-n', type=int, default=50,
                        help='number of epochs to train for. Default=50')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning Rate. Default=1e-3.')
    parser.add_argument('--min-lr', '-mlr', type=float,
                        default=5e-5, help='Minimal Learning Rate. Default=1e-5.')
    parser.add_argument('--ri', type=int, default=5,
                        help='Record Interval. Default=1')
    parser.add_argument('--wd', type=float, default=0,
                        help='Weight Decay. Default=0')
    parser.add_argument('--clip', type=float, default=1e6,
                        help='gradient clipping. Default=1e6')                                             
    parser.add_argument('--init', type=str, default=None,
                        help='which loss to choose. k(kaiming) | x(xavier)', choices=['k', 'x'])
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids')
    parser.add_argument('--no-cuda', action='store_true', help='disable cuda?')
    parser.add_argument('--no-log', action='store_true',
                        help='disable logger?')
    parser.add_argument('--threads', type=int, default=8,
                        help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=2017,
                        help='random seed to use. Default=2017')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--no-ropt', '-nro', action='store_true',
                            help='not resume optimizer')                        
    parser.add_argument('--resumePath', '-rp', type=str,
                        default=None, help='checkpoint to use.')
    parser.add_argument('--test', action='store_true', help='test mode?')
    parser.add_argument('--self-ensemble', action='store_true', help='enable self-ensemble')

    opt = parser.parse_args()
    opt.gpu_ids = _parse_str_args(opt.gpu_ids)
    return opt


class MRAELoss(nn.Module):
    """Mean Relative Absolute Error"""
    def __init__(self, eps=1e-3):
        super(MRAELoss, self).__init__()
        self.eps = eps

    def forward(self, out, gt):
        loss = torch.mean(torch.abs(gt - out) / (gt+self.eps))
        return loss


class SIDLoss(nn.Module):
    """Spectral Information Divergency"""
    def __init__(self, eps=1e-3):
        super(SIDLoss, self).__init__()
        self.eps = eps
    
    def forward(self, p, q):
        N, C, H, W = p.shape
        kl_1 = p * torch.log10((p+self.eps) / (q+self.eps))
        kl_2 = q * torch.log10((q+self.eps) / (p+self.eps))
        sid = kl_1.sum(dim=2).sum(dim=2)
        sid += kl_2.sum(dim=2).sum(dim=2)
        sid = torch.abs(sid)
        sid = torch.sum(sid) / (N * H * W * C)
        return sid


class MultipleLoss(nn.Module):
    def __init__(self, losses, weight=None):
        super(MultipleLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        self.weight = weight or [1/len(self.losses)] * len(self.losses)
    
    def forward(self, predict, target):
        total_loss = 0
        for weight, loss in zip(self.weight, self.losses):
            total_loss += loss(predict, target) * weight
        return total_loss


class Engine(object):
    def __init__(self, opt):
        self.opt = opt
        self.prefix = opt.prefix
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.basedir = None
        self.iterations = None
        self.epoch = None
        self.best_loss = None
        self.writer = None

        self.__setup()

    def __setup(self):
        self.basedir = join('checkpoint', self.opt.arch)
        if not os.path.exists(self.basedir):
            os.mkdir(self.basedir)

        self.best_loss = 1e6
        self.epoch = 0  # start from epoch 0 or last checkpoint epoch
        self.iterations = 0

        cuda = not self.opt.no_cuda
        self.device = 'cuda:%d' %(self.opt.gpu_ids[0]) if cuda else 'cpu'
        #self.device = 'cuda' if cuda else 'cpu'
        print('Cuda Acess: %d' % cuda)
        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        torch.manual_seed(self.opt.seed)
        if cuda:
            torch.cuda.manual_seed(self.opt.seed)

        """Model"""
        print("=> creating model '{}'".format(self.opt.arch))
        self.net = models.__dict__[self.opt.arch]()
        if not self.opt.test: print(self.net)
        # initialize parameters
        if self.opt.init is not None:
            init_params(self.net, init_type=self.opt.init) # disable for default initialization

        if len(self.opt.gpu_ids) > 1:
            self.net = nn.DataParallel(self.net, self.opt.gpu_ids)

        # self.criterion = MRAELoss()
        # self.criterion = SIDLoss()
        # self.criterion = nn.MSELoss(reduce=False)
        # self.criterion = nn.L1Loss(reduce=False)
        self.criterion = MultipleLoss([MRAELoss(), SIDLoss()], [1, 1])

        if cuda:
            self.net.to(self.device)
            self.criterion = self.criterion.to(self.device)

        """Logger Setup"""
        log = not self.opt.no_log
        if log:
            self.writer = get_summary_writer(self.opt.arch)

        """Optimization Setup"""
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.opt.lr, weight_decay=self.opt.wd)

        """Resume previous model"""
        if self.opt.resume:
            # Load checkpoint.
            self.load(self.opt.resumePath, not self.opt.no_ropt)
        else:
            print('==> Building model..')

    def __step(self, train, inputs, targets=None):
        total_norm = None
        if train:
            self.optimizer.zero_grad()
        loss_data = 0

        outputs = self.forward(inputs)

        if targets is None:
            return outputs

        losses = self.criterion(outputs, targets)
        loss = losses.mean()
        if train:                
            loss.backward()
        loss_data += loss.item()
        if train:
            total_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.opt.clip)
            self.optimizer.step()

        return outputs, loss_data, total_norm

    def forward(self, x):
        if self.opt.self_ensemble:
            forward_function = self.net.forward
            return self.forward_x8(x, forward_function)
        else:
            return self.net(x)

    def load(self, resumePath=None, load_opt=True):
        model_best_path = join(self.basedir, self.prefix, 'model_best.pth')
        if os.path.exists(model_best_path):
            best_model = torch.load(model_best_path)
            self.best_loss = best_model['loss']

        print('==> Resuming from checkpoint %s..' % resumePath)
        checkpoint = torch.load(resumePath or model_best_path)
        self.epoch = checkpoint['epoch']
        self.iterations = checkpoint['iterations']
        self.get_net().load_state_dict(checkpoint['net'], strict=False)
        if load_opt:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        
    def validate(self, valid_loader, save=True):
        # se = self.opt.self_ensemble
        # self.opt.self_ensemble = True
        
        self.net.eval()
        validate_loss = 0
        avg_meters = AverageMeters()
        with torch.no_grad():
            for batch_idx, (inputs, targets, _) in enumerate(valid_loader):
                assert inputs.shape[0] == 1
                inputs, targets = inputs.to(self.device), targets.to(self.device)                
                
                # Visualize3D(np.array(inputs))
                # Visualize3D(np.array(targets))
                outputs, loss_data, _ = self.__step(False, inputs, targets)                

                res_dic = MSIQA(outputs, targets)
                # print(res_dic)
                res_dic.update({'Loss': loss_data})
                avg_meters.update(res_dic)

                progress_bar(batch_idx, len(valid_loader), str(avg_meters))

        if not self.opt.no_log:
            write_loss(self.writer, 'validate', avg_meters, self.epoch)

        avg_loss = avg_meters['Loss']
        """Save checkpoint"""
        if avg_loss < self.best_loss and save and not self.opt.no_log:
            print('Best Result Saving...')
            model_best_path = join(self.basedir, self.prefix, 'model_best.pth')
            self.save_checkpoint(
                loss=avg_loss,
                model_out_path=model_best_path                
            )
            self.best_loss = avg_loss

        # self.opt.self_ensemble = se
        return avg_meters, avg_loss
    
    def test(self, test_loader, savedir=None):
        self.net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, _, filenames) in enumerate(test_loader):
                assert inputs.shape[0] == 1
                inputs = inputs.to(self.device)

                outputs = self.__step(False, inputs)
                outputs = quantify_img(outputs)
                print(outputs.shape)
                
                progress_bar(batch_idx, len(test_loader))
                if savedir is not None: 
                    if not os.path.exists(savedir):
                        os.makedirs(savedir)
                    savepath = os.path.join(savedir, filenames[0])
                    from scipy.io import savemat
                    savemat(savepath, {'sr': outputs})

    def save_checkpoint(self, loss, model_out_path=None, **kwargs):
        if not model_out_path:
            model_out_path = join(self.basedir, self.prefix, "model_epoch_%d_%d.pth" % (
                self.epoch, self.iterations))
    
        state = {
            'net': self.get_net().state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': loss,
            'epoch': self.epoch,
            'iterations': self.iterations,
        }
        
        state.update(kwargs)

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if not os.path.isdir(join(self.basedir, self.prefix)):
            os.mkdir(join(self.basedir, self.prefix))

        torch.save(state, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def forward_x8(self, x, forward_function):
        def _transform(v, op):
            
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            
            return ret

        lr_list = [x]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])

        sr_list = [forward_function(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output

    def get_net(self):
        if len(self.opt.gpu_ids) > 1:
            return self.net.module
        else:
            return self.net
