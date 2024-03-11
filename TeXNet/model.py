import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
import os
import torchmetrics
import math
import scipy.io as scio
import segmentation_models_pytorch as smp

DEBUG = True
# 常数
hbar = 105457180e-42
h    = 2*math.pi*hbar
c    = 299792458
kb   = 138064852e-31
e    = 1.60218e-19
cB   = h*c/kb

def dprint(*args, **kwargs):
    # To quickly turn off dprint
    if DEBUG:
        print(*args, **kwargs)

class SMPModel(pl.LightningModule):
    """Class wrapping the TexNet model in PyTorch Lightning ，用 PyTorch Lightning 封装 TexNet 模型的类
    """
    def __init__(self, args):
        super().__init__()
        self.nclass = args.nclass   # number of material classes
        self.eta = 0        # unsupervised的loss权重？
        self.beta = 1
        self.args = args

        self.only_eval = args.eval      # 是否进行评估
        self.timeit = args.timeit       # 是否记录推理时间
        self.log_images = not args.no_log_images    # whether to visualize images on tensorboard
        self.no_T_loss = args.no_T_loss     # 是否计算T、e、v map的loss
        self.no_e_loss = args.no_e_loss
        self.no_v_loss = args.no_v_loss
        self.unsupervised = args.unsupervised   
        self.calc_score = args.calc_score   # whether to train with correlation score or not
        self.checkpoint_dir = args.checkpoint_dir

        # self.slice1 = slice(4, 53)  # for Scene1-10     # Todo
        # self.slice2 = slice(None, None) # for Scene11
        self.slice1 = slice(4, 53, 6) 
        self.slice2 = slice(None, None, 6)
                               #   13 10 6  4  3  2  1
        self.num_inp_ch = 9   #   4  5  9  13 17 25 49      # Todo  # 输入通道数 number_input_channels  

        # normalization values for T and S       T 和 S 的归一化值
        self.mu = torch.tensor([0.12647585, 0.12525924, 0.12395189, 0.12230065, 0.12088306, 0.11962758,
                              0.11836884, 0.11685297, 0.11524992, 0.11388518, 0.11242859, 0.11083422,
                              0.1090912,  0.10737984, 0.10582539, 0.10439677, 0.10263842, 0.10100006,
                              0.0992386,  0.09752469, 0.09576828, 0.09412399, 0.09233064, 0.09060183,
                              0.08907408, 0.08732026, 0.08569063, 0.08377189, 0.08205311, 0.08037362,
                              0.07875945, 0.07714489, 0.07552012, 0.07388812, 0.07219477, 0.07086218,
                              0.06908296, 0.06754399, 0.06604221, 0.06459464, 0.06316591, 0.06165175,
                              0.0602433,  0.05895745, 0.05754419, 0.05616417, 0.05485069, 0.05351864,
                              0.05223851, 0.05066062, 0.0497363,  0.04859088, 0.04738823, 0.04625365])
        self.std = torch.tensor([0.01246481, 0.01251194, 0.0125624,  0.01247964, 0.01251399, 0.01243262,
                               0.0126455,  0.01277499, 0.01247264, 0.01214948, 0.0120328,  0.01196929,
                               0.01211039, 0.01225081, 0.01208897, 0.01186716, 0.01193683, 0.0117601,
                               0.01175319, 0.01168863, 0.01167074, 0.01148603, 0.01150049, 0.01145063,
                               0.0112397,  0.01121394, 0.01108842, 0.01126549, 0.01120692, 0.01110797,
                               0.0109529,  0.01082223, 0.01075425, 0.01073532, 0.01059674, 0.01059848,
                               0.00972673, 0.0094929,  0.00935684, 0.0091823,  0.00900696, 0.00897071,
                               0.00884406, 0.00861178, 0.00857944, 0.00842725, 0.00828631, 0.00812178,
                               0.00806904, 0.00849851, 0.00772755, 0.00772355, 0.00759959, 0.00748127])

        # self.mu = self.mu[:self.num_inp_ch]     # 取前49个？ 不是后49个吗???   self.S_mu = np.reshape(self.S_mu, (-1, 1, 1))[4:53]
        # self.std = self.std[:self.num_inp_ch]

        self.mu = self.mu[self.slice1]       
        self.std = self.std[self.slice1]     
        # self.mu = self.mu[4:53]       
        # self.std = self.std[4:53]     

        self.mu = torch.reshape(self.mu, (-1, 1, 1))    # 转换为三维张量，-1表示通道数与原先一样
        self.std = torch.reshape(self.std, (-1, 1, 1))
        self.T_mu = torch.tensor([15.997467357494212])  # 用于额外的 T 归一化值的张量
        self.T_std = torch.tensor([8.544861474951992])

        # The output of this model is unnormalized logits.
        # We can directly increase the number of channels to train for T and v maps.
        num_out_channels = self.nclass  
        if args.train_T:
            num_out_channels += 1
        if args.train_v:    # sky and ground
            num_out_channels += 2       # 输出通道数  材料种类+T+V*2
        self.train_T = args.train_T
        self.train_v = args.train_v
        if args.no_pretrained:
            # train from scratch
            weight_type = None
        else:
            # weight_type = 'ssl'
            weight_type = 'imagenet'
        self.texnet = getattr(smp, args.model)(encoder_name=args.backbone,      # --backbone resnet50
                                    encoder_weights=weight_type,                # 加载imagenet的预训练权重
                                    in_channels=self.num_inp_ch,          # 输入通道数 49
                                    classes=num_out_channels,             # 输出通道数  材料种类数+ 1(T) + 2(V)
                                )
        # criterion 准则 基准
        self.e_map_criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # 交叉熵loss e-map
        self.T_map_criterion = nn.MSELoss()     # MSE loss  T-map
        self.v_map_criterion = nn.KLDivLoss()   # KL散度 loss v-map
        self.S_pred_criterion = nn.L1Loss()     # L1 loss S_pred
        # Intersection over Union (IoU) 衡量预测分割结果与真实分割结果之间的重叠程度
        self.miou = torchmetrics.JaccardIndex(task="multiclass", num_classes=self.nclass)   
        self.softmax = nn.Softmax(dim=1)

        # if self.nclass == 28:
        #     self.e_val_list = scio.loadmat(os.path.join(args.data_dir, './emiLib.mat'))
        #     self.e_val_list =  self.e_val_list['emiLib'].astype(np.float32)
        # else:
        #     self.e_val_list = scio.loadmat(os.path.join(args.data_dir, f'./emiLib_{self.nclass}.mat'))
        #     self.e_val_list =  self.e_val_list[f'emiLib{self.nclass}'].astype(np.float32)

        ########### Experimental data ################
        self.e_val_list = scio.loadmat(os.path.join(args.data_dir, 'emiLib.mat'))  # material emissivity library 49*30size  每种材料在每个通道上的数据？
        self.e_val_list =  (self.e_val_list['matLib'].astype(np.float32)).transpose()   # 取出'matLib'数据，变为float32数组，转置
        self.e_val_list = torch.from_numpy(self.e_val_list)[:, -self.num_inp_ch:]   # NumPy数组转换为PyTorch张量 

        # self.median_filter = MedianPool2d(stride=1)

    # We cannot override optimizer_zero_grad if accumulate_grad_batches != 1.
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):     # 配置优化器和学习率调度器
        params_group = [
            {'params': self.texnet.parameters(), 'lr': self.args.lr},
        ]

        optimizer = optim.AdamW(params_group, weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[30000, 37000],      # Todo
                                                    gamma=0.1)

        scheduler_dict = {'scheduler': scheduler,
                          'interval': 'epoch',
                          'monitor': 'val_loss'}

        return [optimizer], [scheduler_dict]

    def forward(self, x):
        out = self.texnet(x)

        # Apply median pool to T-map and e-map
        # out[:, :self.nclass] = self.median_filter(out[:, :self.nclass])
        if self.train_T:
            T_pred = out[:, self.nclass:self.nclass+1]
            # out[:, self.nclass:self.nclass+1] = self.median_filter(T_pred)

        return out

    def n2p(self, nv):
        return 1e2*h*c*nv

    def BBn(self, nv,Te):   # nv 频率  Te 温度
        # returns batch x self.num_inp_ch x H x W
        return 2e6*c*nv**2/(torch.exp(cB*1e2*nv/Te)-1)     # 计算黑体辐射

    def BBp(self, nv,Te):  
        # want batch x self.num_inp_ch x 540 x 960
        return self.n2p(nv)*self.BBn(nv,Te)
    
    def unsupervised_S_pred_loss(self,
                                 img,       # torch.Size([1, 49, 256, 256])
                                 T_pred,
                                 e_pred,
                                 v_pred,
                                 S_beta = None,
                                 no_grad = False,
                                 calc_score=True
                                 ):
        # 将张量移动到设备上，以便后续计算
        self.mu = self.mu.to(self.device)
        self.std = self.std.to(self.device)

        self.T_mu = self.T_mu.to(self.device)
        self.T_std = self.T_std.to(self.device)
        # 去标准化
        img = img*self.std + self.mu   
        T_pred = T_pred*self.T_std + self.T_mu
        # pred1 = torch.clamp((torch.nn.functional.relu(pred1)), max=9.1076, min=-4.0624)

        # 这里应该也有错误，49个波段是从760到1240  54个波段是从720到1250
        # nu = np.linspace(720, 1250, self.num_inp_ch)    
        nu = np.linspace(760, 1240, self.num_inp_ch)  
        nu = torch.from_numpy(nu).to(img.device)        # Todo   波段也得改 因为是抽取的  中间少了一些波段
                                                        # 后面直接乘进去了，好像不用改，直接linspace根据self.num_inp_ch分段即可
        batchsize, _, h, w = T_pred.size()      
        emi_val = self.e_val_list[e_pred]
        emi_val = torch.transpose(emi_val, 3, 1)
        # now, emi_val is batch x self.num_inp_ch x W x H
        emi_val = torch.transpose(emi_val, 2, 3)

        # default stride for avg_pool2d is kernel_size
        num_v_channels = v_pred.size(1)
        # if num_v_channels == 2:
        #     quadratic_split = F.avg_pool2d(img, (h//2, w)) # Pass S_beta as an array from file
        # elif num_v_channels == 4:
        #     quadratic_split = F.avg_pool2d(img, (h//2, w//2))
        # else:
        #     raise ValueError(f"number of channels in v cannot be {num_v_channels}")
        # quadratic_split = quadratic_split.reshape(batchsize, self.num_inp_ch, num_v_channels)
        quadratic_split = S_beta
        # 对S进行预测   核心公式！
        S1 = emi_val * self.BBp(nu.reshape(1, self.num_inp_ch, 1, 1), (T_pred+ 273.15)) # 计算自身辐射S1（第一部分）
        v_pred_ = v_pred.reshape(batchsize, num_v_channels, h*w)    
        S2 = torch.matmul(quadratic_split, v_pred_)     # 将调整过后的v_pred与输入的S_beta相乘
        S2 = S2.view(batchsize, self.num_inp_ch, h, w)  # 计算其他物体的散射S2（带有纹理的部分）
        S_pred = S1 + (1-emi_val)*S2                    # 叠加得到物体的热信号S的预测值S_pred  
        S_pred = S_pred.to(img.device)

        img = img.type(torch.float64)

        if no_grad:     # 通常在推理时，不需要跟踪梯度，no_grad可以为True，训练时需要跟踪梯度以便进行反向传播和参数更新
            with torch.no_grad():
                img_ = (img - self.mu)/self.std
                S_pred_ = (S_pred - self.mu)/self.std
                loss = self.S_pred_criterion(img_, S_pred_)     # 通过解耦出T,e,v的预测值，再重新计算S_pred，与原始输入img比较，得到无监督的loss？
        else:                                                   # img是GT吗？ 是GT       # 注意，这是一个解耦问题，从img里面解耦出TeX
            img_ = (img - self.mu)/self.std
            S_pred_ = (S_pred - self.mu)/self.std
            loss = self.S_pred_criterion(img_, S_pred_)
        
        # Correlational loss here.
        EPS = 1e-6
        batch_size, c, h, w = S2.size()

        # Per channel normalization
        img = (img - img.mean(dim=(2, 3), keepdim=True))/(img.std(dim=(2, 3), keepdim=True)+EPS)
        S2 = (S2 - S2.mean(dim=(2, 3), keepdim=True))/(S2.std(dim=(2, 3), keepdim=True)+EPS)

        score = 0.

        if calc_score:
            for b in range(batch_size):
                img_ = img[b]
                S2_ = S2[b]
                # Calculate FFT of S1
                img_FFT = torch.fft.fft2(img_)
                # normalize FFT of S1
                img_FFT_mag = torch.norm(img_FFT, dim=(1, 2), keepdim=True, p='fro')
                # S1_FFT_mag[S1_FFT_mag == 0] = EPS
                img_FFT = img_FFT/img_FFT_mag
                # Calculate inverse FFT
                img_invFFT = torch.real(torch.fft.ifft2(img_FFT))
                score = score + torch.abs(torch.sum(img_invFFT*S2_, dim=(1, 2))).mean()
            
            score = score/batch_size
            # We take log because we want to maximize a positive quantity
            # between 0 and 1. Simply taking the negative may negate the
            # S_pred loss as well.
            score = -torch.log(score)

        loss = loss + score

        return loss, S_pred, score
    
    def pseudo_RGB_from_S_pred(self, S_pred):
        '''
        Convert S_pred to pseudo-RGB for visualization on Tensorboard   把S的预测转换成伪RGB，以便在 Tensorboard 上实现可视化
        '''
        dim4 = True
        if S_pred.dim() == 3:
            # add batch dimension
            S_pred.unsqueeze_(0)
            dim4 = False
        n, c, h, w = S_pred.size()
        
        assert c == self.num_inp_ch

        pool_op = nn.AvgPool3d((4, 1, 1))
        rgb = pool_op(S_pred)
        if not dim4:
            rgb = rgb.squeeze(0)

        return rgb

    def training_step(self, batch, batch_idx):
        self.e_val_list = self.e_val_list.to(self.device)
        S_beta, img, tgt = batch
        # e may be from 1 to 21 (both inclusive).
        # So we need to correct it, if that is the case.
        t, e, v = tgt
        num_v_channels = v.size(1)
        # e = e.unsqueeze(1)
        # e -= torch.min(e) ## Cropping issue?
        t = t.unsqueeze(1)
        tgt = (t, e, v)
        out = self(img)

        self.train_img = img
        self.train_out = out
        self.train_tgt = tgt

        T_pred = None
        v_pred = None
        e_pred = out[:, :self.nclass, :, :]

        if self.no_e_loss:
            with torch.no_grad():
                loss_e = self.e_map_criterion(e_pred, e)
        else:
            loss_e = self.e_map_criterion(e_pred, e)

        loss_T = torch.tensor([0.])
        loss_v = torch.tensor([0.])
        if self.train_T:
            T_pred = out[:, self.nclass:self.nclass+1, :, :]
            if self.no_T_loss:
                with torch.no_grad():
                    loss_T = self.T_map_criterion(T_pred, t)
            else:
                loss_T = self.T_map_criterion(T_pred, t)
        if self.train_v:
            # last 4 channels, because we are not sure if T is also
            # predicted or not.
            v_pred = out[:, -num_v_channels:, :, :]
            v_pred = self.softmax(v_pred)
            EPS = 1e-4
            v_pred_log = torch.log(v_pred+EPS)
            if self.no_v_loss:
                with torch.no_grad():
                    loss_v = self.v_map_criterion(v_pred_log, v)
            else:
                loss_v = self.v_map_criterion(v_pred_log, v)
        
        e_pred = torch.argmax(e_pred, 1, keepdim=False).type(torch.long)

        if T_pred is None:
            T_pred = t
        if v_pred is None:
            v_pred = v

        if not self.unsupervised:
            # no_grad不计算梯度，只需要执行前向传播并获取输出结果，通常用于推理或评估，不需要保存中间计算结果以进行反向传播。
            with torch.no_grad():       # supervised 使用no_grad，loss_S在后面没用到
                loss_S, S_pred, corr_score = self.unsupervised_S_pred_loss(img, T_pred, e_pred, v_pred,S_beta,
                                                                            no_grad=True, calc_score=self.calc_score)
        else:   # unsupervised 不使用no_grad，loss_S在后面按照权重加到了loss里
            loss_S, S_pred, corr_score = self.unsupervised_S_pred_loss(img, T_pred, e_pred, v_pred, S_beta,
                                                                        calc_score=self.calc_score)
        # save S_pred for predicting later
        self.train_S_pred = S_pred

        loss1 = 0
        loss = 0

        if not self.no_e_loss:
            loss1 = loss1+loss_e
        if not self.no_T_loss and self.train_T:
            loss1 = loss1+loss_T
        if not self.no_v_loss and self.train_v:
            loss1 = loss1+loss_v

        if self.unsupervised:
            loss = (1-self.eta)*loss1+(self.eta)*(self.beta)*loss_S     # beta是1，eta是各种loss的权重？文章里是0.5
        else:
            loss = loss1     # loss1是supervised的loss(由预测的T_pred,e_pred,v_pred和GT得到的loss)
                             # loss_S是unsupervised的loss（通过预测的S_pred与真实值img计算得到的loss）
                             # S_pred也由T_pred,e_pred,v_pred计算得到
        train_step_out = {'loss': loss}

        try:
            for group in self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups:
                current_lr = group['lr']
                break
            self.log('lr', current_lr, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)
        except:
            pass

        self.log('loss', loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=False)
        self.log('loss1', loss1, on_step=True, on_epoch=False, sync_dist=True, prog_bar=False)
        self.log('eta', self.eta, on_step=True, on_epoch=False, sync_dist=True, prog_bar=False)

        # logging individual losses
        self.log('loss_T', loss_T, on_step=True, on_epoch=False, sync_dist=True, prog_bar=False)
        self.log('loss_e', loss_e, on_step=True, on_epoch=False, sync_dist=True, prog_bar=False)
        self.log('loss_v', loss_v, on_step=True, on_epoch=False, sync_dist=True, prog_bar=False)
        self.log('loss_S', loss_S, on_step=True, on_epoch=False, sync_dist=True, prog_bar=False)
        self.log('corr_score', corr_score, on_step=True, on_epoch=False, sync_dist=True, prog_bar=False)

        # # 用这个print看--eval的时候有没有经过training_step   没有   --ava的时候只eval  不train
        # print("training...__________________________________________________________________________________________")

        with torch.no_grad():
            train_iou = self.miou(e_pred, e)
            self.log('train_miou', train_iou, on_step=True, on_epoch=False, sync_dist=True, prog_bar=False)

        return train_step_out

    def training_epoch_end(self, outputs):
        T, e, v = tuple([_.detach() for _ in self.train_tgt])
        e = e.unsqueeze(1)
        # self.eta += (1./self.trainer.max_epochs)
        # self.eta = float((self.current_epoch/self.trainer.max_epochs)**3)
        self.eta = 0.5 # Only unsupervised loss
        # print("Eta = ", self.eta)

        num_v_channels = v.size(1)

        out = self.train_out.detach()

        '''
        self.train_S_pred is not normalized but self.train_img is
        normalized. To calculate the error for visualization, use error
        between normalized values because that is the actual error we
        are using for backpropagation. But for visualization on
        tensorboard, de-normalize the values.
        '''
        S_pred_ = self.train_S_pred.detach()
        S_pred_ = (S_pred_ - self.mu)/self.std  
        S_true_ = self.train_img.detach()
        loss_S = torch.abs(S_pred_ - S_true_).mean() ###### Write this to file
        
        with torch.no_grad():
            self.train_img = self.train_img * self.std + self.mu  

        S_pred = self.pseudo_RGB_from_S_pred(self.train_S_pred.detach())    # 预测值
        S_true = self.pseudo_RGB_from_S_pred(self.train_img.detach())       # GT
        S_error = self.pseudo_RGB_from_S_pred(torch.abs(self.train_img.detach()- \
                                                self.train_S_pred.detach()))   
        e_pred = out[:, :self.nclass, ...]
        T_pred = None
        v_pred = None
        loss_v = torch.tensor([0.])
        if self.train_T:
            T_pred = out[:, self.nclass:self.nclass+1, ...]
        if self.train_v:
            v_pred = out[:, -num_v_channels:, ...]
            v_pred = self.softmax(v_pred)
            loss_v = self.v_map_criterion(torch.log(v_pred), v)
        
        loss_e = self.e_map_criterion(F.softmax(e_pred, 1), e.squeeze(1))
        e_pred = torch.argmax(e_pred, 1, keepdim=True)

        self.T_mu = self.T_mu.to(self.device)
        self.T_std = self.T_std.to(self.device)
        loss_T = torch.tensor([0.])
        if self.train_T:
            er1 = torch.abs(T_pred-T)*self.T_std
            loss_T = er1.mean()
        er2 = (e != e_pred).type(torch.int)
        if self.train_v:
            er3 = torch.abs(v_pred-v)

        num_samples = min(2, e_pred.size(0))
        idx = np.random.choice(e_pred.size(0), size=num_samples, replace=False)
        idx = torch.from_numpy(idx)

        S_pred = S_pred[idx]
        S_error = S_error[idx]
        S_true = S_true[idx]

        if self.logger and self.log_images:         # --no_log_images == True   self.log_images == False 不画图像
            self.logger.experiment.add_images('S_pred_train', S_pred, self.current_epoch, dataformats='NCHW')
            self.logger.experiment.add_images('S_error_train', S_error, self.current_epoch, dataformats='NCHW')
            self.logger.experiment.add_images('S_true_train', S_true, self.current_epoch, dataformats='NCHW')

        er2 = er2[idx]
        e_pred = e_pred[idx]
        e = e[idx]
        if self.logger and self.log_images:
            self.logger.experiment.add_images('predicted_e_train', e_pred, self.current_epoch, dataformats='NCHW')
            self.logger.experiment.add_images('er_e_train', er2, self.current_epoch, dataformats='NCHW')
            self.logger.experiment.add_images('GT_e_train', e, self.current_epoch, dataformats='NCHW')

        if self.train_T:
            er1 = er1[idx]
            T_pred = T_pred[idx]
            T = T[idx]
            T_pred = (T_pred*self.T_std)+self.T_mu
            T = (T*self.T_std)+self.T_mu
            T = torch.tile(T, (1, 3, 1, 1))
            T_pred = torch.tile(T_pred, (1, 3, 1, 1))
            er1 = torch.tile(er1, (1, 3, 1, 1))
            if self.logger and self.log_images:
                self.logger.experiment.add_images('predicted_T_train', T_pred, self.current_epoch, dataformats='NCHW')
                self.logger.experiment.add_images('er_T_train', er1, self.current_epoch, dataformats='NCHW')
                self.logger.experiment.add_images('GT_T_train', T, self.current_epoch, dataformats='NCHW')

        if self.train_v:
            idx = np.random.choice(er3.size(0))
            er3 = er3[idx].squeeze().unsqueeze(1)
            v_pred = v_pred[idx].squeeze().unsqueeze(1)
            v = v[idx].squeeze().unsqueeze(1)
            v_pred = torch.tile(v_pred, (1, 3, 1, 1))
            v = torch.tile(v, (1, 3, 1, 1))
            er3 = torch.tile(er3, (1, 3, 1, 1))
            if self.logger and self.log_images:
                self.logger.experiment.add_images('predicted_v_train', v_pred, self.current_epoch, dataformats='NCHW')
                self.logger.experiment.add_images('er_v_train', er3, self.current_epoch, dataformats='NCHW')
                self.logger.experiment.add_images('GT_v_train', v, self.current_epoch, dataformats='NCHW')
        
        # write loss_S to a file for later visualization
        with open(os.path.join(self.checkpoint_dir, 'loss_train.txt'), 'a') as write_S_error:
            write_S_error.write(f"{self.current_epoch},{loss_S.item()},{loss_T.item()}"+ \
                                    f",{loss_e.item()},{loss_v.item()}\n")

        return

    def validation_step(self, batch, batch_idx):
        self.e_val_list = self.e_val_list.to(self.device)
        S_beta, img, tgt = batch        #  解压批次数据
        # e may be from 1 to 21 (both inclusive).
        # So we need to correct it, if that is the case.
        t, e, v = tgt
        num_v_channels = v.size(1)
        # e -= torch.min(e) ## Cropping issue? 
        t = t.unsqueeze(1)  # 对t维度扩展，并更新target
        tgt = (t, e, v)

        with torch.no_grad():
            pred = self.texnet(img)     # 对img进行预测，并将预测结果分割为e_pred、T_pred和v_pred
            e_pred = pred[:, :self.nclass, :, :]
            T_pred = None
            v_pred = None
            if self.train_T:
                # using 20:21 instead of 20 will keep dimensions.
                T_pred = pred[:, self.nclass:self.nclass+1, :, :]
            else:
                T_pred = t
            if self.train_v:
                # last 4 channels, because we are not sure if T is also
                # predicted or not.
                v_pred = pred[:, -num_v_channels:, :, :]
                v_pred = self.softmax(v_pred)
                v_pred_log = torch.log(v_pred)
            else:
                v_pred = v
            loss_T = torch.tensor([0.])
            loss_v = torch.tensor([0.])
            loss_e = self.e_map_criterion(e_pred, e)
            if self.train_T:
                loss_T = self.T_map_criterion(T_pred, t)
            if self.train_v:
                loss_v = self.v_map_criterion(v_pred_log, v)

            # loss = loss_T + loss_e + loss_v
        
            e_pred = torch.argmax(e_pred, 1, keepdim=False).type(torch.long)
            val_iou = self.miou(e_pred, e)
            loss_S, S_pred, corr_score = self.unsupervised_S_pred_loss(img, T_pred, e_pred, v_pred, S_beta,
                                                                        no_grad=True, calc_score=self.calc_score)
            # _, SSS_true, _ = self.unsupervised_S_pred_loss(img, t, e, v, S_beta,
            #                                                             no_grad=True, calc_score=self.calc_score)
        # 计算loss
        loss1 = 0
        loss = 0

        if not self.no_e_loss:
            loss1 = loss1+loss_e
        if not self.no_T_loss and self.train_T:
            loss1 = loss1+loss_T
        if not self.no_v_loss and self.train_v:
            loss1 = loss1+loss_v

        if self.unsupervised:
            loss = (1-self.eta)*loss1+(self.eta)*(self.beta)*loss_S
        else:
            loss = loss1
        # loss存于dict
        val_step_out = dict()

        val_step_out['loss'] = loss
        val_step_out['loss1'] = loss1
        val_step_out['loss_T'] = loss_T
        val_step_out['loss_e'] = loss_e
        val_step_out['val_iou'] = val_iou
        val_step_out['loss_v'] = loss_v
        val_step_out['loss_S'] = loss_S
        val_step_out['corr_score'] = corr_score
        # val_step_out['pred'] = pred.detach().cpu()
        # val_step_out['S_pred'] = S_pred.detach().cpu()
        # val_step_out['S_true'] = img.detach().cpu()
        # tgt = [_.detach().cpu() for _ in tgt]
        # val_step_out['pred'] = pred.detach()
        # val_step_out['S_pred'] = S_pred.detach()
        # val_step_out['S_true'] = img.detach()
        # tgt = [_.detach() for _ in tgt]
        # val_step_out['tgt'] = tgt

        # print("Validation about to save tensors")

        if self.only_eval:
            torch.save(t, f'{self.checkpoint_dir}/val_T_{batch_idx}.pt')       # 把t,e,v,pred,S_pred,img(true)数据保存至.pt
            torch.save(e, f'{self.checkpoint_dir}/val_e_{batch_idx}.pt')
            torch.save(v, f'{self.checkpoint_dir}/val_v_{batch_idx}.pt')
            torch.save(pred, f'{self.checkpoint_dir}/val_pred_{batch_idx}.pt')
            torch.save(S_pred, f'{self.checkpoint_dir}/val_S_pred_{batch_idx}.pt')
            torch.save(img, f'{self.checkpoint_dir}/val_S_true_{batch_idx}.pt')

        # print("Validation tensors")

        # print("Validation step end")
        return val_step_out

    def validation_epoch_end(self, outputs):
        # if outputs and self.only_eval:
        #     pred = [out['pred'] for out in outputs]
        #     tgt = [out['tgt'] for out in outputs]
        #     S_pred = [out['S_pred'] for out in outputs]
        #     S_true = [out['S_true'] for out in outputs]
        #     T = [i[0] for i in tgt]
        #     e = [i[1] for i in tgt]
        #     v = [i[2] for i in tgt]

        #     T = torch.cat(T, 0)
        #     e = torch.cat(e, 0).unsqueeze(1)
        #     v = torch.cat(v, 0)

        #     pred = torch.cat(pred, 0)
        #     S_pred = torch.cat(S_pred, 0)
        #     S_true = torch.cat(S_true, 0)

            # print("Validation about to save tensors")

            # torch.save(T, f'{self.checkpoint_dir}/val_T.pt')
            # torch.save(e, f'{self.checkpoint_dir}/val_e.pt')
            # torch.save(v, f'{self.checkpoint_dir}/val_v.pt')
            # torch.save(pred, f'{self.checkpoint_dir}/val_pred.pt')
            # torch.save(S_pred, f'{self.checkpoint_dir}/val_S_pred.pt')
            # torch.save(S_true, f'{self.checkpoint_dir}/val_S_true.pt')

            # print("Validation tensors")

        if outputs and not self.only_eval:
            # pred = [out['pred'] for out in outputs]
            # tgt = [out['tgt'] for out in outputs]
            # S_pred = [out['S_pred'] for out in outputs]
            # S_true = [out['S_true'] for out in outputs]
            # T = [i[0] for i in tgt]
            # e = [i[1] for i in tgt]
            # v = [i[2] for i in tgt]

            # T = torch.cat(T, 0)
            # e = torch.cat(e, 0)
            # v = torch.cat(v, 0)
            # num_v_channels = v.size(1)

            # pred = torch.cat(pred, 0)
            # S_pred = torch.cat(S_pred, 0)
            # S_true = torch.cat(S_true, 0)

            # S_error = torch.abs(S_pred-S_true)
            # loss_S = S_error.mean() ####### Write this to file
            # # Convert to pseudo-RGB for visualization
            # S_pred = self.pseudo_RGB_from_S_pred(S_pred)
            # S_true = self.pseudo_RGB_from_S_pred(S_true)
            # S_error = self.pseudo_RGB_from_S_pred(S_error)
            # e_pred = pred[:, :self.nclass, ...]
            # loss_e = torch.tensor([0.])
            # loss_e = self.e_map_criterion(e_pred, e) ######### Write this to file
            # e_pred = torch.argmax(e_pred, 1)
            # er2 = (e != e_pred).type(torch.int)

            # num_samples = min(2, e_pred.size(0))
            # idx = np.random.choice(e_pred.size(0), size=num_samples, replace=False)
            # idx = torch.from_numpy(idx)

            # S_pred = S_pred[idx]
            # S_error = S_error[idx]
            # S_true = S_true[idx]

            # if self.logger and self.log_images:
            #     self.logger.experiment.add_images('S_pred_val', S_pred, self.current_epoch, dataformats='NCHW')
            #     self.logger.experiment.add_images('S_error_val', S_error, self.current_epoch, dataformats='NCHW')
            #     self.logger.experiment.add_images('S_true_val', S_true, self.current_epoch, dataformats='NCHW')

            # er2 = er2[idx]
            # e_pred = e_pred[idx]
            # e = e[idx]

            # # for visualization purposes
            # e.unsqueeze_(1)
            # e_pred.unsqueeze_(1)
            # er2.unsqueeze_(1)
            # e = torch.tile(e, (1, 3, 1, 1))
            # e_pred = torch.tile(e_pred, (1, 3, 1, 1))
            # er2 = torch.tile(er2, (1, 3, 1, 1))
            # if self.logger and self.log_images:
            #     self.logger.experiment.add_images('predicted_e_val', e_pred, self.current_epoch, dataformats='NCHW')
            #     self.logger.experiment.add_images('error_e_val', er2, self.current_epoch, dataformats='NCHW')
            #     self.logger.experiment.add_images('GT_e_val', e, self.current_epoch, dataformats='NCHW')

            # T_pred = None
            # loss_T = torch.tensor([0.])
            # if self.train_T:
            #     T_pred = pred[:, self.nclass:self.nclass+1, ...]
            #     # self.T_mu = self.T_mu.to("cpu")
            #     # self.T_std = self.T_std.to("cpu")
            #     self.T_mu = self.T_mu.to("cpu")
            #     self.T_std = self.T_std.to("cpu")
            #     er1 = torch.abs(T-T_pred)*self.T_std
            #     loss_T = er1.mean() ########### Write this to file
            #     T_pred = (T_pred*self.T_std) + self.T_mu
            #     T = (T*self.T_std) + self.T_mu
            #     T_pred = T_pred[idx]
            #     T = T[idx]
            #     er1 = er1[idx]

            #     # for visualization purposes
            #     T = torch.tile(T, (1, 3, 1, 1))
            #     T_pred = torch.tile(T_pred, (1, 3, 1, 1))
            #     er1 = torch.tile(er1, (1, 3, 1, 1))
            # v_pred = None
            # loss_v = torch.tensor([0.])
            # if self.train_v:
            #     v_pred = pred[:, -num_v_channels:, ...]
            #     v_pred = self.softmax(v_pred)
            #     loss_v = self.v_map_criterion(torch.log(v_pred), v)
            #     er3 = torch.abs(v_pred-v)
            #     # We have space to visualize only sample from the batch
            #     idx = np.random.choice(v_pred.size(0))
            #     v_pred = v_pred[idx].squeeze().unsqueeze(1)
            #     v = v[idx].squeeze().unsqueeze(1)
            #     er3 = er3[idx].squeeze().unsqueeze(1)
            #     v_pred = torch.tile(v_pred, (1, 3, 1, 1))
            #     v = torch.tile(v, (1, 3, 1, 1))
            #     er3 = torch.tile(er3, (1, 3, 1, 1))
            #     if self.logger and self.log_images:
            #         self.logger.experiment.add_images('predicted_v_val', v_pred, self.current_epoch, dataformats='NCHW')
            #         self.logger.experiment.add_images('error_v_val', er3, self.current_epoch, dataformats='NCHW')
            #         self.logger.experiment.add_images('GT_v_val', v, self.current_epoch, dataformats='NCHW')
            #     if self.logger and self.log_images:
            #         self.logger.experiment.add_images('predicted_T_val', T_pred, self.current_epoch, dataformats='NCHW')
            #         self.logger.experiment.add_images('error_T_val', er1, self.current_epoch, dataformats='NCHW')
            #         self.logger.experiment.add_images('GT_T_val', T, self.current_epoch, dataformats='NCHW')

            # v_pred = None
            # loss_v = torch.tensor([0.])
            # if self.train_v:
            #     v_pred = pred[:, -num_v_channels:, ...]
            #     v_pred = self.softmax(v_pred)
            #     loss_v = self.v_map_criterion(torch.log(v_pred), v)
            #     er3 = torch.abs(v_pred-v)
            #     # We have space to visualize only sample from the batch
            #     idx = np.random.choice(v_pred.size(0))
            #     v_pred = v_pred[idx].squeeze().unsqueeze(1)
            #     v = v[idx].squeeze().unsqueeze(1)
            #     er3 = er3[idx].squeeze().unsqueeze(1)
            #     v_pred = torch.tile(v_pred, (1, 3, 1, 1))
            #     v = torch.tile(v, (1, 3, 1, 1))
            #     er3 = torch.tile(er3, (1, 3, 1, 1))
            #     if self.logger and self.log_images:
            #         self.logger.experiment.add_images('predicted_v_val', v_pred, self.current_epoch, dataformats='NCHW')
            #         self.logger.experiment.add_images('error_v_val', er3, self.current_epoch, dataformats='NCHW')
            #         self.logger.experiment.add_images('GT_v_val', v, self.current_epoch, dataformats='NCHW')

            avg_loss = torch.tensor([out['loss'] for out in outputs], device=self.device).mean()
            self.log('val_loss', avg_loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
            avg_e_loss = torch.tensor([out['loss_e'] for out in outputs], device=self.device).mean()
            self.log('val_loss_e', avg_e_loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
            miou = torch.tensor([out['val_iou'] for out in outputs], device=self.device).mean()
            self.log('val_miou', miou, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
            if self.train_T:
                avg_T_loss = torch.tensor([out['loss_T'] for out in outputs], device=self.device).mean()
                self.log('val_loss_T', avg_T_loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
            if self.train_v:
                avg_v_loss = torch.tensor([out['loss_v'] for out in outputs], device=self.device).mean()
                self.log('val_loss_v', avg_v_loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
            avg_S_loss = torch.tensor([out['loss_S'] for out in outputs], device=self.device).mean()
            self.log('val_loss_S', avg_S_loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
            avg_corr_score = torch.tensor([out['corr_score'] for out in outputs], device=self.device).mean()
            self.log('val_corr_score', avg_corr_score, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)

            # # write losses to a file for later visualization
            # with open(os.path.join(self.checkpoint_dir, 'loss_val.txt'), 'a') as write_S_error:
            #     write_S_error.write(f"{self.current_epoch},{loss_S.item()},{loss_T.item()}"+\
            #                         f",{loss_e.item()},{loss_v.item()}\n")
        
        # Print all val values so that we can quickly see it in the
        # terminal itself.
        print("\n Validation results\n")
        print("Loss: ", np.mean([out['loss'].item() for out in outputs]))
        print("Loss1", np.mean([out['loss1'].item() for out in outputs]))
        print("Loss_T: ", np.mean([out['loss_T'].item() for out in outputs]))
        print("Loss_e: ", np.mean([out['loss_e'].item() for out in outputs]))
        print("Loss_v: ", np.mean([out['loss_v'].item() for out in outputs]))
        print("Loss_S: ", np.mean([out['loss_S'].item() for out in outputs]))
        print("Val_iou: ", np.mean([out['val_iou'].item() for out in outputs]))
        print("Corr_score: ", np.mean([out['corr_score'] for out in outputs]))

        # If we also want to time the inference, we do it separately.
        if self.only_eval and self.timeit:
            total_time = 0.
            max_iter = 100
            h, w = 1080, 1920
            for it in range(max_iter):
                rand_inp = torch.rand(1, self.num_inp_ch, h, w, device=self.device)
                start_time = time.time()
                _ = self.texnet(rand_inp)
                total_time += time.time() - start_time
            
            print("Average inference time", total_time/max_iter)

        return
