import argparse
import glob, os
from collections import OrderedDict
from train import *

cv2.setNumThreads(0)    # fix hanging problem

parser = argparse.ArgumentParser()

# training & testing loop settings
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--max_epoch', type=int, default=30, help='Number of epochs for training')
parser.add_argument('--pretrain_epoch', type=int, default=5)
parser.add_argument('--resume_epoch', type=int, default=10, help='Default is 0, change if want to resume')
parser.add_argument('--resume_id', type=int, default=0, help='Default is 0, change if want to resume')
parser.add_argument('--batch', type=int, default=8, help='Training batch size')

parser.add_argument('--val_percent', type=float, default=10.0)

parser.add_argument('--model_name', type=str, default='delse')
parser.add_argument('--exp_name', type=str, default='checkpoints')          # folder name of experiment results
parser.add_argument('--txt', type=str, default='')                  # assign a suffix for a specific experiment setting

parser.add_argument('--test_interval', type=int, default=5, help='Run on test set every args.test_interval epochs')
parser.add_argument('--save_freq', type=int, default=5, help='Store a model every args.  save_freq epochs')
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--e2e', type=bool, default=False)

# optimizer settings
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--adjust_epoch', type=int, nargs='+', default=(10, 20, 30))
parser.add_argument('--lr_decay', type=float, default=0.3)
parser.add_argument('--wd', type=float, default=1e-6, help='Weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--ave_grad', type=int, default=1, help='Average the gradient of several iterations')

# loss & model settings
parser.add_argument('--backend_cnn', type=str, default='resnet101-skip4',
                    choices=['resnet101', 'resnet101-skip4'])
parser.add_argument('--concat_dim', type=int, default=128, choices=[64, 128], help='concat dim of skip features')
parser.add_argument('--resolution', type=int, default=256, help='resolution of input images')

parser.add_argument('--loss', type=str, default='ac', choices=['ce','bd','ac','surface'])
parser.add_argument('--alpha', type=float, default=1, help='loss ratios')
parser.add_argument('--gamma', type=float, default=0, help='loss ratios')
parser.add_argument('--loss_weight', type=float, default=0.01, help='altra loss ratios')
parser.add_argument('--epsilon', type=float, default=-1, help='param for Heaviside function')

parser.add_argument('--T', type=int, default=0)
parser.add_argument('--timestep', type=int, default=5)      # timestep can't be too large
parser.add_argument('--classifier', type=str, default='psp', choices=['atrous', 'psp'])

# data settings
parser.add_argument('--dataset', type=str, default='refuge',
                    choices=['refuge', 'prostate', 'dgs'])
parser.add_argument('--cmap', action='store_true', default=True)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--dt_max', type=float, default=30)
parser.add_argument('--input_channels', type=int, default=4, help='Number of input channels')
parser.add_argument('--relax_crop', type=int, default=10, help='Enlarge the bounding box by args.relax_crop pixels')
parser.add_argument('--zero_pad_crop', type=bool, default=True)
parser.add_argument('--mask_threshold', type=float, default=-2)


def check_args():
    args = parser.parse_args()
    assert args.resume_epoch < args.max_epoch

    # experiment root dir
    args.save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.exp_name)
    if not os.path.exists(args.save_dir_root):
        os.makedirs(args.save_dir_root)

    # experiment dir
    runs = sorted(glob.glob(os.path.join(args.save_dir_root, 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
    args.save_dir = os.path.join(args.save_dir_root, 'run_%04d' % run_id)
    if not os.path.exists(os.path.join(args.save_dir, 'models')):
        os.makedirs(os.path.join(args.save_dir, 'models'))

    return args


def generate_param_report(args, model):
    log_path = os.path.join(args.save_dir, args.exp_name + '.txt')

    f = open(log_path, 'w')
    attrs = dir(args)
    for attr in attrs:
        if attr[0] == '_':
            continue
        v = getattr(args, attr)
        f.write(attr + ': ' + str(v) + '\n')

    p = OrderedDict()
    if args.mode == 'train':
        p['dataset_train'] = str(model.trainset)
        p['transformations_train'] = [str(t) for t in model.composed_transforms_tr.transforms]
        p['dataset_test'] = str(model.valset)
        p['transformations_test'] = [str(t) for t in model.composed_transforms_ts.transforms]
        p['optimizer'] = str(model.optimizer)
    elif args.mode == 'test':
        p['dataset_test'] = str(model.valset)
        p['transformations_test'] = [str(t) for t in model.composed_transforms_ts.transforms]
        p['optimizer'] = str(model.optimizer)
    for k, v in p.items():
        f.write(k + ': ' + str(v) + '\n')

    f.close()


def train():
    args = check_args()
    model = DELSE(args)
    generate_param_report(args, model)
    for epoch in range(args.resume_epoch, args.max_epoch):
        # adjust lr
        model.adjust_learning_rate(epoch, adjust_epoch=args.adjust_epoch, ratio=args.lr_decay)
        # train one epoch
        model.train(epoch)
        # checkpoints
        if (epoch % args.save_freq) == args.save_freq - 1 and epoch != 0:
            model.save_ckpt(epoch)
        # test one epoch
        if args.test_interval >= 0 and epoch % args.test_interval == (args.test_interval - 1):
            print('Testing')
            model.test(epoch, _save=False)

def test():
    model = DELSE(args)
    generate_param_report(args, model)
    model.test(args.resume_epoch - 1, _save=True)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()




