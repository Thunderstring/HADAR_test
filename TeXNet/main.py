import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin

import torch
import numpy as np
import os
import json

from config import parse_args
from model import SMPModel
# from datamodule import HADARLoader
from datamodule_newdata import HADARMultipleScenesLoader


# python TeXNet/main.py --ngpu 1 --backbone resnet50 --data_dir ~/workspace/zx/HADAR_database/ --workers 8 --epochs 40000 --checkpoint_dir supervised_crop --lr 1e-3 --weight-decay 1e-3 --train_T --train_v --no_log_images --eval_every 500 --res full --batch-size 10 --seed 42 --unsupervised --fold 4
# python TeXNet/main.py --ngpu 1 --backbone resnet50 --data_dir ~/workspace/zx/HADAR_database/ --workers 8 --epochs 40000 --checkpoint_dir supervised_crop --lr 1e-3 --weight-decay 1e-3 --train_T --train_v --no_log_images --eval_every 500 --res full --batch-size 10 --seed 42
# python TeXNet/main.py --ngpu 1 --backbone resnet50 --data_dir ~/workspace/zx/HADAR_database/ --workers 8 --epochs 200 --checkpoint_dir supervised_crop --lr 1e-3 --weight-decay 1e-3 --train_T --train_v --eval_every 10 --res full --batch-size 10 --seed 42
# python main.py --ngpus 1 --backbone resnet50 --data_dir ../ --workers 8 --epochs 40000 --checkpoint_dir supervised_crop --lr 1e-3 --weight-decay 1e-3 --train_T --train_v --no_log_images --eval_every 500 --res full --batch-size 10 --seed 42
# val python TeXNet/main.py --ngpus 1 --backbone resnet50 --data_dir ~/workspace/zx/HADAR_database/ --workers 8 --epochs 40000 --checkpoint_dir supervised_check_crop --lr 1e-3 --weight-decay 1e-3 --train_T --train_v --no_log_images --eval_every 500 --res full --batch-size 10 --seed 42 --resume ~/workspace/zx/HADAR-main/supervised_crop_v1/lightning_logs/version_1/checkpoints/epoch=36499-step=109500.ckpt --eval

if __name__ == "__main__":
    args = parse_args()     # 解析命令行参数

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)

    model_checkpoint = True
    if args.checkpoint_dir == '' or args.checkpoint_dir is None:
        args.checkpoint_dir = 'tmp_ckpt'
        model_checkpoint = False

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if model_checkpoint:
        logger = TensorBoardLogger(save_dir=args.checkpoint_dir,
                                   version=1,
                                   name='lightning_logs')
    else:
        logger = None
    
    overfit_batches = 0
    if args.overfit:
        overfit_batches = 2
    
    # AMP is Automatic Mixed Precision.
    # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
    if args.use_amp:        # 自动混合精度（Automatic Mixed Precision）
        precision = 16 # bits
    else:
        precision = 32
    
    callback_list = []

    if model_checkpoint:        # 创建了一个 ModelCheckpoint 回调对象，使用验证集的loss，保存最佳的两个模型，保存最后一个模型
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=2, save_last=True)
        callback_list.append(checkpoint_callback)

    plugins_list = []
    
    model = SMPModel(args)      # 实例化模型，传入args
    # datamodule = HADARLoader(args)
    datamodule = HADARMultipleScenesLoader(args)    # 数据模块

    if args.ngpus <= 1:
        sync_bn = False
    else:
        sync_bn = True
    
    if args.swa:
        from pytorch_lightning.callbacks import StochasticWeightAveraging as SWA
        swa = SWA(swa_lrs=1e-3)
        callback_list.append(swa)

    trainer = pl.Trainer(
                        #  devices=[1],   # 如果要用卡1
                         devices=args.ngpus,
                         strategy='ddp_find_unused_parameters_false',   # 分布式训练策略设置为 DistributedDataParallel（DDP），确保训练过程中不会忽略未使用的参数。
                         accelerator="gpu",
                         plugins=plugins_list,
                         num_nodes=args.num_nodes,  # 分布式训练的节点数
                         amp_backend='native',
                         auto_lr_find=True,     # 启用自动学习率查找功能，有助于确定模型训练过程中的最佳学习率
                         benchmark=True, # only if the input sizes don't change rapidly 基准模式，如果输入大小保持不变，可以提高训练性能。
                         callbacks=callback_list,
                         default_root_dir=args.checkpoint_dir,
                         fast_dev_run=args.quick_check,
                         gradient_clip_val=args.grad_clip, # 0 gradient clip means no clipping
                         logger=logger,
                         check_val_every_n_epoch=args.eval_every,   # 训练过程中多久进行一次验证，每 args.eval_every 个 epoch 进行一次
                         max_epochs=args.epochs,
                         overfit_batches=overfit_batches,       # 训练过程中用于过拟合的批次数量
                         precision=precision, # decides the use of AMP
                         sync_batchnorm=sync_bn,
                         detect_anomaly=True,
                         enable_progress_bar=True       # 启用进度条
                         )
    
    # dump training options to a JSON file.
    if model_checkpoint:
        json_dump_dict = vars(args)
        with open(os.path.join(args.checkpoint_dir, 'training_args.json'), 'w') as json_f:
            json.dump(json_dump_dict, json_f)
        print(f"** Dumped training arguments to {args.checkpoint_dir}/training_args.json")

    if args.eval:       # 如果需要进行模型评估，则加载训练模型并执行验证
        model = SMPModel.load_from_checkpoint(args.resume, args=args)
        trainer.validate(model, datamodule=datamodule)
    else:
        if args.resume != "":       # 如果有预训练模型，从与训练模型开始训练
            trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)
        else:
            trainer.fit(model, datamodule=datamodule)
