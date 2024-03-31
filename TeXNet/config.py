import configargparse
import os
import datetime
import shutil
import torch

def parse_args():
    p = configargparse.ArgParser(default_config_files=['configs/default_config.ini'])
    p.add('-c', is_config_file=True,        # 使用configargparse库创建了一个参数解析器 p
            help='config file path')        # 在命令行中使用 -c 参数时，会将其解析为配置文件
    
    # model
    p.add_argument('--model', type=str, default='PAN',  # 模型名称
                        help='model name (default: pan)')
    p.add_argument('--backbone', type=str, default='resnet50',  # 主干网络
                        help='backbone name (default: resnet50)')
    p.add_argument("--no_pretrained", action="store_true",  
                        help="whether to start training from scratch")

    # datasets
    p.add_argument('--dataset', type=str, default='hadar',
                        help='dataset name (default: hadar)')
    p.add_argument('--workers', type=int, default=8,        # 数据加载线程数
                        metavar='N', help='dataloader threads')     
    p.add_argument('--base-size', type=int, default=520,
                        help='base image size')
    p.add_argument('--crop-size', type=int, default=None,
                        help='crop image size')
    p.add_argument('--data_dir', type=str, required=True,
                        help='location to data directory')
    p.add_argument('--batch-size', type=int, default=1,     # 训练和测试的批次大小
                        metavar='N', help='input batch size for \
                        training (default: auto)')
    p.add_argument("--randerase", action="store_true",
                        help="whether to use random erasure or not")    # 是否随机擦除
    p.add_argument("--res", default='half', type=str,       # 使用完整或半分辨率数据
                        help="use data at full/half resolution (default: half)")    
    p.add_argument('--nclass', type=int, default=30, # 6 for exp
                        help='number of material classes')      # 材料类别数
    p.add_argument("--eval_on_train", action="store_true",
                        help="whether to evaluate on the training data.")
    p.add_argument('--num_train', type=int, default=-1,
                        help='number of training points to use (-1 means all)')

    # training hyper params     训练超参数
    p.add_argument('--num_nodes', type=int, default=1,
                        help='number of available nodes/systems')
    p.add_argument('--ngpus', type=int, default=1,
                        help='number of available GPUs')
    p.add_argument('--aux', action='store_true', default= False,
                        help='Auxilary Loss')       # 辅助loss
    p.add_argument('--aux-weight', type=float, default=0.2,
                        help='Auxilary loss weight (default: 0.2)')
    p.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: auto)')
    p.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    p.add_argument('--test-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                        testing (default: same as batch size)')
    p.add_argument('--lambda1', type=float, default=8e-3,
                        help='lambda1 (for Temperature)')   # TeX的lambda
    p.add_argument('--lambda2', type=float, default=1.,
                        help='lambda2 (for Emissivity)')
    p.add_argument('--lambda3', type=float, default=2.,
                        help='lambda3 (for Texture)')
    p.add_argument('--eval_every', type=int, default=10,
                        metavar='N', help='validate every __ epochs')

    # optimization params       优化超参数
    p.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: auto)')
    p.add_argument('--lr-scheduler', type=str, default='poly',      # 学习率调度器，如何调整学习率的策略
                        help='learning rate scheduler (default: poly)')
    p.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    p.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 1e-4)')
    p.add_argument('--use_amp', action='store_true', default= False,
                        help='whether to use AMP or not')
    p.add_argument('--grad_clip', type=float, default=0,
                        help='value at which gradients would be clipped. 0 means "no gradient clipping"')
    p.add_argument('--dropprob', type=float, default=0.1,
                        help='Dropout/Droppath probability')
    p.add_argument("--swa", action="store_true",
                        help="whether to use SWA or not")
    p.add_argument("--use_kldiv", action="store_true",     # v-map是否使用KL散度（KL Divergence），KL 散度通常用于度量两个概率分布之间的差异
                        help="whether to use KL div for v-map or not")
    p.add_argument("--train_T", action="store_true",
                        help="whether to train for T-map or not")       # 是否训练T、v map
    p.add_argument("--train_v", action="store_true",
                        help="whether to train for v-map or not")
    p.add_argument("--no_v_loss", action="store_true",
                        help="whether to remove loss for v-map or not")
    p.add_argument("--no_T_loss", action="store_true",
                        help="whether to remove loss for T-map or not")
    p.add_argument("--no_e_loss", action="store_true",
                        help="whether to remove loss for e-map or not")
    p.add_argument("--unsupervised", action="store_true",
                        help="whether to train unsupervisedly or not")  # 无监督 不用标签，用文中给的材料库、plank's law 和数学结构
    p.add_argument("--only_S_loss", action="store_true",
                        help="only use loss_S")

    # checkpoint
    p.add_argument('--resume', type=str, default='',
                        help='put the path to resuming file if needed') # 如果需要，输入恢复文件的路径
    p.add_argument('--checkname', type=str, default='default',
                        help='set the checkpoint name')
    p.add_argument('--checkpoint_dir', type=str, default='',
                        help='name of the directory where checkpoints would be saved')
    p.add_argument('--model-zoo', type=str, default=None,
                        help='evaluating on model zoo model')
    p.add_argument("--fold", default=None, type=int,
                        help="fold number for cross validation")        # 交叉验证的fold数 

    # evaluation option
    p.add_argument('--eval', action='store_true', default= False,
                        help='run only evaluation')
    p.add_argument('--timeit', action='store_true', default= False,
                        help='get the inference time')      # 获取推理时间，用于评估模型的速度
    p.add_argument('--test-val', action='store_true', default= False,
                        help='generate masks on val set')
    p.add_argument('--no-val', action='store_true', default= False,
                        help='skip validation during training')

    # multi grid dilation option
    p.add_argument("--multi-grid", action="store_true", default=False,
                        help="use multi grid dilation policy")
    p.add_argument('--multi-dilation', nargs='+', type=int, default=None,
                        help="multi grid dilation list")
    p.add_argument('--os', type=int, default=8,
                        help='output stride default:8')

    # misc
    p.add_argument("--quick_check", action="store_true",
                        help="whether to do a quick check or not")
    p.add_argument("--overfit", action="store_true",
                        help="whether to overfit on a small subset of data or not")
    p.add_argument("--no_log_images", action="store_true",
                        help="whether to visualize images on tensorboard or "+\
                            "not (to avoid large log files)")   # 是否在 TensorBoard 上可视化图像，以避免生成大型日志文件
    p.add_argument("--calc_score", action="store_true",
                        help="whether to train with correlation score or not")
    p.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    p.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    p.add_argument("--show_all_gpu_outputs", action="store_true",
                        help="whether to show outputs from all GPUs or not")
    args = p.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        args.epochs = 250
    if args.lr is None:
        args.lr = 1e-3
    
    # Save the source code as a zip file    接下来的代码段用于将源代码文件保存为一个 ZIP 文件
    now = datetime.datetime.now()
    ts = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(now.year, now.month, now.day, now.hour, now.minute,
                                                            now.second)

    # Get your current location
    src_dir = os.getcwd()
    # Path to save the source code
    checkpoint_dir = args.checkpoint_dir
    # Name of zip folder
    src_name = "src"
    # Create a temporary folder to keep the source
    dst_dir = os.path.join(checkpoint_dir, src_name)

    os.makedirs(dst_dir, exist_ok=True)

    curr_dirname = os.getcwd().split('/')[-1]
    # file formats that you want
    file_formats = ('.txt', '.py', '.ini')

    # Create a new folder, and copy the required files to it.  创建新文件夹，把需要的文件放进去(txt,py,ini)
    for root, subdirs, files in os.walk(src_dir):
        for f in files:
            if f.endswith(file_formats) and checkpoint_dir not in root:
                root_ = root.split(curr_dirname)[1][1:]
                src_path = os.path.join(root, f)
                dst_path = os.path.join(dst_dir, root_)
                os.makedirs(dst_path, exist_ok=True)
                # Copy the required files
                shutil.copy(src=src_path, dst=dst_path)

    # Write the time stamp
    with open(os.path.join(dst_dir, 'timestamp'), 'w') as f:
        f.write(ts+'\n')

    # Zip the folder
    shutil.make_archive(dst_dir, format='zip', root_dir=checkpoint_dir, base_dir=src_name)
    # Remove the temporary folder
    shutil.rmtree(dst_dir)

    return args
