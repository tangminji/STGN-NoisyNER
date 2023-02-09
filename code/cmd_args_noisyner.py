import argparse
from common.losses import *
from experimentalsettings import ExperimentalSettings
import datetime
import logging
import utils
import torch

# Settings
parser = argparse.ArgumentParser(description='NoisyNER')

parser.add_argument('--exp_name', default='noisyner', type=str)
parser.add_argument('--sub_script', default='sbatch_noisyner_sub.sh', type=str)
parser.add_argument('--out_tmp', default='noisyner_out_tmp.json', type=str)
parser.add_argument('--params_path', default='noisyner_params.json', type=str)
parser.add_argument('--log_dir', default='ner/', type=str)

parser.add_argument('--dataset', default='NoisyNER', type=str, help="Model type selected in the list: [MNIST, CIFAR10, CIFAR-100, CIFAR-10_5K, UTKFACE]")
parser.add_argument('--loss', default='CE', type=str, help="loss type")
parser.add_argument('--num_class', default=4, type=int) # noisy_ner是四分类
#parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train')#300
#parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_sig', type=float, default=0.005, help='learning rate for sigma iteration')
parser.add_argument('--noise_mode', type=str, default='dependent', help='Noise mode in the list: [sym, asym, dependent]')
parser.add_argument('--noise_rate', type=float, default=0.4, help='Noise rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--forget_times', type=int, default=1, help='thereshold to differentiate clean/noisy samples')
parser.add_argument('--num_gradual', type=int, default=0, help='epochs for warmup')
parser.add_argument('--ratio_l', type=float, default=0.5, help='element1 to total ratio')
parser.add_argument('--total', type=float, default=1.0, help='total amount of every elements')
parser.add_argument('--patience', type=int, default=3, help='patience for increasing sig_max for avoiding overfitting')
parser.add_argument('--times', type=float, default=3.0, help='increase perturb by times')
parser.add_argument('--avg_steps', type=int, default=10, help='step nums at most to calculate k1')
parser.add_argument('--adjustimes', type=int, default=10, help='Maximum number of adjustments')
parser.add_argument('--sigma', type=float, default=0.05, help='STD of Gaussian noise')#label0.5/para5e-3/moutput5e-3
parser.add_argument('--sig_max', type=float, default=0.1, help='max threshold of sigma')
parser.add_argument('--smoothing', type=float, default=0.1, help='used in mode Label_smoothing')
parser.add_argument('--delay_eps', type=float, default=50.0, help='p-norm of adaptive regularization')
parser.add_argument('--early_eps', type=float, default=200.0, help='p-norm of adaptive regularization')
parser.add_argument('--pnorm', type=float, default=2.0, help='p-norm of adaptive regularization')
parser.add_argument('--beta', type=float, default=0.9, help='beta for exponential moving average of the gradient')
parser.add_argument('--skip_clamp_param', default=False, const=True, action='store_const',
                    help='Do not clamp data parameters during optimization')
parser.add_argument('--gpu_id', type=int, default=0, help='index of gpu to use')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--restart', default=True, const=True, action='store_const',
                    help='Erase log and saved checkpoints and restart training')#False
parser.add_argument('--mode', type=str, default='no_GN',
                    choices=['GN_on_label',
                             'GN_on_moutput',
                             'GN_on_parameters',
                             'no_GN',
                             'GN_noisy_samples',
                             'GN_gods_perpective2',
                             'GN_gods_perpective3',
                             'Random_walk'])#'GN_on_marchitecture'，'GN_on_feaemb'


# IO
parser.add_argument('--input_dir', type=str, default='../preprocessed',
                    help='dir that contains the dataset')
parser.add_argument('--output_dir', type=str, required=True,
                    help='root directory that contains images')
parser.add_argument('--word_embedding_path', type=str, default="../data/fasttext/cc.et.300.bin",
                    help='FastText embedding path')

# data reading
parser.add_argument('--data_separator', type=str, default="\t",
                    help='input format: token[data_separator]label')
parser.add_argument('--label_format', default='io', choices=['io', 'bio'],
                    help='io or bio format')


# experiment related
parser.add_argument('--label_set', type=int, required=True, choices=[0, 1, 2, 3, 4, 5, 6, 7])
parser.add_argument('--num_times', type=int, default=100,
                    help='triple of (start stop step_size)')
parser.add_argument('--ns', nargs='+', type=int, default=[], help='number of samples for each label')
parser.add_argument('--uniform_sampling', action='store_true',
                    help='if yes, then we sample the clean data randomly, '
                            'otherwise we sample the equal size of data points per label')
parser.add_argument('--exp_settings', type=str, default='ni_exp', help='a list of experiment settings')
parser.add_argument('--train_settings', type=str, default='base_with_noise', help=' a list training settings')
parser.add_argument('--random_seed', type=int, default=5555, help='random_seed')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers')

# Training related arguments
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch_size for training')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning_rate')

args = parser.parse_args()

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# 根据exp_name修正参数， 这样可以减少要传入参数的数量
if 'STGN' in args.exp_name:
    args.mode = 'Random_walk'
if 'GCE' in args.exp_name:
    args.loss = 'GCE'
if 'SLN' in args.exp_name:
    args.mode = 'GN_on_label'


# read experiment settings/configs
exp_setting_name = args.exp_settings
train_setting_name = args.train_settings



staring_time = datetime.datetime.now()
staring_time_str = staring_time.strftime("%b_%d_%H_%M_%S")

logger, log_dir = utils.create_logger(args.output_dir, args, staring_time)

logger.info(f"EXP={exp_setting_name}, TRAIN={train_setting_name} begins")

EXP_SETTINGS = ExperimentalSettings.load_json(exp_setting_name, logger, dir_path="./exp_config/")
EXP_SETTINGS["LABEL_SET"] = args.label_set
EXP_SETTINGS["OUTPUT_DIR"] = args.output_dir
if args.ns != []:
    EXP_SETTINGS["NS"] = args.ns

TRAIN_SETTINGS = ExperimentalSettings.load_json(train_setting_name, logger)

args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


CIFAR10_CONFIG = {
    "CE": CELoss(reduction='none'),
    "FL": FocalLoss(gamma=0.5),
    "MAE": MAELoss(num_classes=10, reduction='none'),
    "GCE": GCELoss(num_classes=10, q=0.01, reduction='none'),
    "SCE": SCELoss(num_classes=10, a=0.1, b=1),
    # "NLNL": NLNL(train_loader, num_classes=10),
    "NFL": NormalizedFocalLoss(gamma=0.5, num_classes=10),
    "NGCE": NGCELoss(num_classes=10),
    "NCE": NCELoss(num_classes=10),
    "NFL+RCE": NFLandRCE(alpha=1, beta=1, num_classes=10, gamma=0.5),
    "NCEandMAE": NCEandMAE(alpha=1, beta=1, num_classes=10),
    "NCEandRCE": NCEandRCE(alpha=1, beta=1, num_classes=10),
}

CIFAR100_CONFIG = {
    "CE": nn.CrossEntropyLoss(),
    "FL": FocalLoss(gamma=0.5),
    "MAE": MAELoss(num_classes=100),
    "GCE": GCELoss(num_classes=100, q=0.001),
    "SCE": SCELoss(num_classes=100, a=6, b=0.1),
    # "NLNL": NLNL(train_loader, num_classes=10),
    "NFL": NormalizedFocalLoss(gamma=0.5, num_classes=100),
    "NGCE": NGCELoss(num_classes=100),
    "NCE": NCELoss(num_classes=100),
    "NFL+RCE": NFLandRCE(alpha=10, beta=1, num_classes=100, gamma=0.5),
    "NCEandMAE": NCEandMAE(alpha=10, beta=1, num_classes=100),
    "NCEandRCE": NCEandRCE(alpha=10, beta=1, num_classes=100),
}

#learning_rate_schedule = np.array([80, 100, 160])#for CIFAR10/CIFAR100

