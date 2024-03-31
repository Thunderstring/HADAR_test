import os
import numpy as np
import scipy.io as scio
from tqdm import tqdm
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

class HADARMultipleScenes():
    NUM_CLASS = 30
    """docstring for HADARSegmentation"""

    def __init__(self, root='~/workspace/zx/HADAR_database/',
                 split='train', inp_transform=None, target_transform=None,
                 **kwargs):     # **kwargs表示用kwargs接收所有参数，并存储成dict

        root = os.path.expanduser(root)     # os.path.expanduser() 函数用于展开路径中的波浪号 ~，将其替换为当前用户的主目录路径
        # root = os.path.join(root, "HADAR_database")
        self.randflip = kwargs.get('randflip', False) and split == "train"      # split代表数据种类?(train, val or test)
        fold = kwargs.get('fold', None)

        print("Fold is", fold, "for split", split)   # 第几个文件夹用来train, val or test
        # we manually split the database, instead of randomly splitting, to ensure the same diversity of the validation set and training set
        if fold is None:        # train_all
            train_ids = ["L_0001", "L_0002", "L_0003", "L_0004", "L_0005", "R_0001", "R_0002", "R_0003", "R_0004", "R_0005"]
            val_ids = ["L_0001", "R_0001"] # one fresh sample and one sample from training set.
            test_ids = ["L_0001", "R_0001"]  

            train_exp_ids = ["0001", "0002", "0003", "0004"]
            val_exp_ids = ["0001"] 
            test_exp_ids = ["0001"]
        elif fold == 0:
            train_ids = ["L_0002", "L_0003", "L_0004", "L_0005"]
            train_ids += ["R_0002", "R_0003", "R_0004", "R_0005"]
            val_ids = ["L_0001", "R_0001"]
            test_ids = ["L_0001", "R_0001"]

            train_exp_ids = ["0002", "0003", "0004"]
            val_exp_ids = ["0001"] 
            test_exp_ids = ["0001"]
        elif fold == 1:
            train_ids = ["L_0001", "L_0003", "L_0004", "L_0005"]
            train_ids += ["R_0001", "R_0003", "R_0004", "R_0005"]
            val_ids = ["L_0002", "R_0002"]
            test_ids = ["L_0002", "R_0002"]

            train_exp_ids = ["0001", "0003", "0004"]
            val_exp_ids = ["0002"]
            test_exp_ids = ["0002"]
        elif fold == 2:
            train_ids = ["L_0001", "L_0002", "L_0004", "L_0005"]
            train_ids += ["R_0001", "R_0002", "R_0004", "R_0005"]
            val_ids = ["L_0003", "R_0003"]
            test_ids = ["L_0003", "R_0003"]

            train_exp_ids = ["0001", "0002", "0004"]
            val_exp_ids = ["0003"]
            test_exp_ids = ["0003"]
        elif fold == 3:
            train_ids = ["L_0001", "L_0002", "L_0003", "L_0005"]
            train_ids += ["R_0001", "R_0002", "R_0003", "R_0005"]
            val_ids = ["L_0004", "R_0004"]
            test_ids = ["L_0004", "R_0004"]

            train_exp_ids = ["0001", "0002", "0003"]
            val_exp_ids = ["0004"]
            test_exp_ids = ["0004"]
        elif fold == 4:
            train_ids = ["L_0001", "L_0002", "L_0003", "L_0004"]
            train_ids += ["R_0001", "R_0002", "R_0003", "R_0004"]
            val_ids = ["L_0005", "R_0005"]
            test_ids = ["L_0005", "R_0005"]

            train_exp_ids = ["0001", "0002", "0003"]
            val_exp_ids = ["0004"]
            test_exp_ids = ["0004"]
        elif fold == 5:  
            # Scene11
            val_ids = None                       
            val_exp_ids = ["0001", "0002", "0003", "0004"] 
        elif fold == 6: 
            # Scene1-10       
            val_ids = ["L_0001", "L_0002", "L_0003", "L_0004", "L_0005", "R_0001", "R_0002", "R_0003", "R_0004", "R_0005"]
            val_exp_ids = None


        if split == 'train':
            ids = train_ids
            exp_ids = train_exp_ids
        elif split == 'val':
            ids = val_ids
            exp_ids = val_exp_ids
        elif split == 'test':
            ids = test_ids
            exp_ids = test_exp_ids

        print('IDs for', split, 'are', ids, 'for synthetic data and', exp_ids, 'for experimental data')

        # SUBFOLDERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]      # 读取Scene
        SUBFOLDERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]          # Todo
        # SUBFOLDERS = [1,2,3,4,5]
        # SUBFOLDERS = [11]
        SUBFOLDERS = ["Scene"+str(_) for _ in SUBFOLDERS]
        # 文件名列表，用于后续load数据
        self.S_files = []       
        self.S_beta_files = []
        self.T_files = []
        self.e_files = []
        self.v_files = []

        for subfolder in SUBFOLDERS:    # 把每个Scene的heatcube.mat EnvObj.npy 存入相应的列表
            # Scene 11 has only 4 frames, unlike other scenes which have 5 frames.
            if subfolder == "Scene11":
                ids_ = exp_ids
            else:
                ids_ = ids
            for id in ids_:
                ######### Synthetic ####################        heatvube, tMap, vMap has been changed from .mat to .npy
                self.S_files.append(os.path.join(root, subfolder, 'HeatCubes',   
                                                 f"{id}_heatcube.npy"))
                                                #  f"{id}_heatcube.mat"))     # 原始数据 观测到的HeatCubes, heatcube.mat
                self.S_beta_files.append(os.path.join(root, subfolder, 'HeatCubes',  
                                                 'S_EnvObj_'+f"{id}.npy"))      # S_beta,显著物体辐射 S_EnvObj_.npy
                self.T_files.append(os.path.join(root, subfolder, 'GroundTruth',
                                                 'tMap', f"tMap_{id}.npy"))
                                                #  'tMap', f"tMap_{id}.mat"))     # GT, tmap_.mat
                self.e_files.append(os.path.join(root, subfolder, 'GroundTruth',
                                                 'eMap', f"new_eMap_{id}.npy")) # GT, emap_.npy
                self.v_files.append(os.path.join(root, subfolder, 'GroundTruth',
                                                 'vMap', f"vMap_{id}.npy"))
                                                #  'vMap', f"vMap_{id}.mat"))     # GT, vmap_.mat

        self.inp_transforms = inp_transform # transform for inputs
        self.tgt_transforms = target_transform # transform for target values
        self.split = split      # train, val or test

        self.num_points = len(self.S_files)         # heatcubes 数量

        ######################### Synthetic data ###############################   
        # 高光谱数据heatcubes中每个通道的均值和标准差
        self.channels = 6      # 共54个channels 合成后分别有 54 27 18 9 6   Todo
        # 54channels
        if self.channels == 54:
            self.S_mu = np.array([0.12647585, 0.12525924, 0.12395189, 0.12230065, 0.12088306, 0.11962758,
                                0.11836884, 0.11685297, 0.11524992, 0.11388518, 0.11242859, 0.11083422,
                                0.1090912,  0.10737984, 0.10582539, 0.10439677, 0.10263842, 0.10100006,
                                0.0992386,  0.09752469, 0.09576828, 0.09412399, 0.09233064, 0.09060183,
                                0.08907408, 0.08732026, 0.08569063, 0.08377189, 0.08205311, 0.08037362,
                                0.07875945, 0.07714489, 0.07552012, 0.07388812, 0.07219477, 0.07086218,
                                0.06908296, 0.06754399, 0.06604221, 0.06459464, 0.06316591, 0.06165175,
                                0.0602433,  0.05895745, 0.05754419, 0.05616417, 0.05485069, 0.05351864,
                                0.05223851, 0.05066062, 0.0497363,  0.04859088, 0.04738823, 0.04625365])
            self.S_std = np.array([0.01246481, 0.01251194, 0.0125624,  0.01247964, 0.01251399, 0.01243262,
                                0.0126455,  0.01277499, 0.01247264, 0.01214948, 0.0120328,  0.01196929,
                                0.01211039, 0.01225081, 0.01208897, 0.01186716, 0.01193683, 0.0117601,
                                0.01175319, 0.01168863, 0.01167074, 0.01148603, 0.01150049, 0.01145063,
                                0.0112397,  0.01121394, 0.01108842, 0.01126549, 0.01120692, 0.01110797,
                                0.0109529,  0.01082223, 0.01075425, 0.01073532, 0.01059674, 0.01059848,
                                0.00972673, 0.0094929,  0.00935684, 0.0091823,  0.00900696, 0.00897071,
                                0.00884406, 0.00861178, 0.00857944, 0.00842725, 0.00828631, 0.00812178,
                                0.00806904, 0.00849851, 0.00772755, 0.00772355, 0.00759959, 0.00748127])
        # 27 channels
        elif self.channels == 27:
            self.S_mu = np.array([0.25375898, 0.24832648, 0.24260284, 0.23721034, 0.23114588, 0.22530844,
                                    0.21845065, 0.21213148, 0.20553776, 0.19864764, 0.19172371, 0.18475876,
                                    0.17817069, 0.17124736, 0.16421847, 0.15764397, 0.1511071, 0.14460252,
                                    0.13819623, 0.13218129, 0.12631956, 0.12068867, 0.11510446, 0.10972008,
                                    0.10433387, 0.09964647, 0.09494889])
            self.S_std = np.array([0.02558138, 0.0256097, 0.02552195, 0.02586953, 0.02517336, 0.02458021,
                                    0.02480102, 0.02439279, 0.02420689, 0.02396575, 0.02364765, 0.02341166,
                                    0.02294647, 0.02279733, 0.02277475, 0.02225276, 0.02192101, 0.02151032,
                                    0.01994221, 0.01929764, 0.01871723, 0.01815256, 0.01766579, 0.01708387,
                                    0.01700574, 0.01604973, 0.01565284])
        # 18channels
        elif self.channels == 18:
            self.S_mu = np.array([0.3787343, 0.365954, 0.35346534, 0.34019932, 0.32524871, 0.31087118,
                                    0.29534224, 0.27978787, 0.26473776, 0.24889875, 0.23401641, 0.21933717,
                                    0.20502046, 0.19167661, 0.1789374,  0.16657582, 0.15471751, 0.14421172])
            self.S_std = np.array([0.03843782, 0.03830342, 0.03859022, 0.03703229, 0.03709133, 0.03631913,
                                    0.03587734, 0.03514529, 0.03427457, 0.0342493,  0.0332321,  0.032431,
                                    0.02967465, 0.02827346, 0.02705211, 0.02584295, 0.02503429, 0.02367499])
        # 9channels
        elif self.channels == 9:
            self.S_mu = np.array([0.7446883, 0.69366466, 0.63611989, 0.57513011, 0.51363652, 0.45335359,
                                    0.39669708, 0.34551321, 0.29892923])
            self.S_std = np.array([0.07662647, 0.07553818, 0.07336723, 0.07101703, 0.06848451, 0.06562432,
                                    0.05793278, 0.05288228, 0.04867232])
        # 6 channels
        elif self.channels == 6:
            self.S_mu = np.array([1.09815364, 0.97631921, 0.83986788, 0.70225234, 0.57563447, 0.46550505])
            self.S_std = np.array([0.1151695, 0.11031681, 0.10522565, 0.09981513, 0.08497367, 0.0744513])

        self.T_mu = 15.997467357494212
        self.T_std = 8.544861474951992
        #############################################################################
        ############################ Experimental data ##############################

        # self.slice1 = slice(4,53)  # for Scene1-10      # Todo
        self.slice1 = slice(None, None)   # for only Scene1-10  54channels for integral

        self.slice2 = slice(None, None) # for Scene11

        # self.slice1 = slice(4, 53, 6)           
        # self.slice2 = slice(None, None, 6)

        # 将self.S_mu转换为形状(54，1，1)，然后对其进行切片,取4到52，49个通道
        self.S_mu = np.reshape(self.S_mu, (-1, 1, 1))[self.slice1]       
        self.S_std = np.reshape(self.S_std, (-1, 1, 1))[self.slice1]     
        # self.S_mu = np.reshape(self.S_mu, (-1, 1, 1))[4:53]       
        # self.S_std = np.reshape(self.S_std, (-1, 1, 1))[4:53]   

        self._load_data() # Loads data to CPU

    def _load_data(self):
        # load the data location into two variables and return them
        self.S_beta = []    # 显著物体环境的辐射        49*2
        self.S = []         # Heatcube 原始数据        49*1080*1920
        self.tMaps = []     # GT, tmap                1080*1920
        self.eMaps = []     # GT, emap                1080*1920
        self.vMaps = []     # GT, vmap                [2, 1080, 1920]

        for i in self.S_beta_files:     # Scene1-10: 1*54*2*1   Scene11: 1*49*2*1
            data = np.load(i)
            data = np.squeeze(data)     # 压缩数据的单维度条目

            if data.shape[0] == 54:
                data = data[self.slice1]         
            else : 
                data = data[self.slice2]        
            # if data.shape[0] == 54:
            #     data = data[4:53]         
            # else : 
            #     data = data       

            data = torch.from_numpy(data).type(torch.float)
            self.S_beta.append(data)

        for i in self.S_files:          # Scene1-10: 1080*1920*54   Scene11: 260*1500*49
            # data = scio.loadmat(i)        # ??    表示已被修改
            # if "S" in data.keys():
            #     data = data["S"]        # 'S' for Scene1-10
            # elif "HSI" in data.keys():
            #     data = data["HSI"]      # 'HSI' for Scene11 
            # else:
            #     raise ValueError("Known keys not present in heatcubes")
            data = np.load(i)

            data = np.transpose(data, (2, 0, 1))  # 转置 (0,1,2)-->(2,0,1)  即把通道数放在最前面data.shape[0]==channels

            if data.shape[0]==54:
                data = data[self.slice1]     
            else : 
                data = data[self.slice2]        
            # if data.shape[0]==54:
            #     data = data[4:53]     
            # else : 
            #     data = data      

            data = (data-self.S_mu)/self.S_std      # 数据预处理，标准化，减均值，除标准差 计算一下heatcube的均值和标准差，看看和给的对不对
            # data = data[4:53]
            # data = data[4:53] # 49 channels
            # print("Shape of S = ", np.shape(data))
            data = torch.from_numpy(data).type(torch.float)     # 转换为torch.float，便于后续计算
            self.S.append(data)

        for i in self.T_files:          # Scene1-10: 1080*1920   Scene11: 260*1500
            # data = scio.loadmat(i)["tMap"]      # ??
            data = np.load(i)
            data = (data-self.T_mu)/self.T_std
            data = torch.from_numpy(data).type(torch.float)
            self.tMaps.append(data)

        for i in self.e_files:          # Scene1-10: 1080*1920   Scene11: 260*1500
            # data = scio.loadmat(i)["eMap"]
            # data = torch.from_numpy(data).type(torch.long)
            # self.eMaps.append(data)
            # # print("Shape of e = ", np.shape(data))

            ##### Synthetic data ####
            data = np.load(i)
            data = torch.from_numpy(data).type(torch.long)
            self.eMaps.append(data)

        for i in self.v_files:          # Scene1-10: 1080*1920*2   Scene11: 260*1500*2
            # data = scio.loadmat(i)["vMap"]      # ??
            data = np.load(i)
            data = np.transpose(data, (2, 0, 1))
            data = torch.from_numpy(data).type(torch.float)
            self.vMaps.append(data)

        # self.S_beta = torch.stack(self.S_beta)
        # self.S = torch.stack(self.S)
        # self.tMaps = torch.stack(self.tMaps)
        # self.eMaps = torch.stack(self.eMaps)
        # self.vMaps = torch.stack(self.vMaps)

    def __len__(self):
        if self.split == "train":
            return self.num_points
        else:
            return 2*self.num_points

    def __getitem__(self, index_):      # __getitem__(self,key)方法返回所给键对应的值  
                                        # 此例中根据index_返回S_beta和随机裁剪后的 S, (tMap, eMap, vMap)
        if self.split == 'train':
            # Ensure all samples are seen during training.
            index = index_
            size_idx = 0
        else:       # ？
            index = index_ // 2 # choose the scene and the frame
            size_idx = index_ % 2 # choose the size/crop of the frame
            
        S_beta = self.S_beta[index]
        S = self.S[index]
        tMap = self.tMaps[index]
        eMap = self.eMaps[index]
        vMap = self.vMaps[index]

        if self.inp_transforms is not None:
            S = self.inp_transforms(S)

        if self.tgt_transforms is not None:
            tMap = self.tgt_transforms(tMap)
            eMap = self.tgt_transforms(eMap)
            vMap = self.tgt_transforms(vMap)

        if self.split == 'train' or (self.split == "val" and size_idx == 0):
            ### Arbitrary Crop ####   # Since the real-world scene (260*1500) has a different image size with the synthetic scenes (1080*1920)
            crop_size = 256           # we used random crop (256*256) in training.
            w, h = S.shape[1:]      # 获取weight和height
            th, tw = 256, 256 ## Hardcoding crop sizes

            x1 = random.randint(0, w - tw)  # 生成随机裁剪起始点(x1, y1)，防止裁剪区域超过图像尺寸
            y1 = random.randint(0, h - th)

            S.to(torch.float32)
            tMap.to(torch.float32)
            vMap.to(torch.float32)
            eMap.to(torch.float32)
            # print("Avgs before =", torch.mean(S), torch.mean(tMap), torch.mean(vMap), torch.mean(eMap))

            S = transforms.functional.resized_crop(S, x1, y1, th, tw, crop_size)

            tMap = torch.unsqueeze(tMap, 0)
            tMap = transforms.functional.resized_crop(tMap, x1, y1, th, tw, crop_size)
            tMap = torch.squeeze(tMap, 0)

            vMap = transforms.functional.resized_crop(vMap, x1, y1, th, tw, crop_size)

            eMap = torch.unsqueeze(eMap, 0)
            eMap = transforms.functional.resized_crop(eMap, x1, y1, th, tw, crop_size)
            eMap = torch.squeeze(eMap, 0)

            S.to(torch.float32)
            tMap.to(torch.float32)
            vMap.to(torch.float32)
            eMap.to(torch.long)

            if self.randflip:       # 随机翻转
                flip = torch.rand(1) > 0.5
                if flip:
                    S = TF.hflip(S)
                    tMap = TF.hflip(tMap)
                    eMap = TF.hflip(eMap)
                    vMap = TF.hflip(vMap)

            ### Fixed Crop ####

            # print("Avgs after =", torch.mean(S), torch.mean(tMap), torch.mean(vMap), torch.mean(eMap))
            # w, h = S.shape[1:]
            # th, tw = 256, 256 ## Hardcoding crop sizes

            # x1 = random.randint(0, w - tw)
            # y1 = random.randint(0, h - th)
            # S = transforms.functional.crop(S, x1,y1, th, tw)
            # tMap = transforms.functional.crop(tMap, x1,y1, th, tw)
            # vMap = transforms.functional.crop(vMap, x1,y1, th, tw)
            # eMap = transforms.functional.crop(eMap, x1,y1, th, tw)

            # if self.randflip:
            #     flip = torch.rand() > 0.5
            #     if flip:
            #         # S_beta = TF.hflip(S_beta)
            #         S = TF.hflip(S)
            #         tMap = TF.hflip(tMap)
            #         eMap = TF.hflip(eMap)
            #         vMap = TF.hflip(vMap)

            ## OLD datamodule ###

        # if self.randflip:
        #     flip = torch.rand() > 0.5
        #     if flip:
        #         S = TF.hflip(S)
        #         tMap = TF.hflip(tMap)
        #         eMap = TF.hflip(eMap)
        #         vMap = TF.hflip(vMap)

        # print("Shapes = ", S_beta.shape, S.shape, tMap.shape, eMap.shape, vMap.shape)

        return S_beta, S, (tMap, eMap, vMap)

class HADARMultipleScenesLoader(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.data_dir = args.data_dir
        self.args = args # for later use
        self.num_workers = args.workers
        self.dataset_name = args.dataset
        self.dataset_type = HADARMultipleScenes
        self.randerase = args.randerase
        self.randflip = args.randerase # for now, we will tie those options together.
        self.eval_on_train = args.eval_on_train
        self.fold = args.fold
        self.eval_only = args.eval

    def setup(self, stage=None):
        # input 
        train_inp_transform_list = [] #[transforms.RandomCrop(self.args.crop_size)]
        if self.randerase and False:
            train_inp_transform_list.extend([transforms.RandomErasing(p=0.5, scale=(0.1, 0.5))])

        if len(train_inp_transform_list) > 0:
            train_inp_transform = transforms.Compose(train_inp_transform_list)     # 把由转换操作组成的列表组合成一个整体的转换操作
        else:
            train_inp_transform = None
        # target
        train_tgt_transform = None

        if not self.eval_only:      # 如果不是eval模式，就train 
            print(f"** Loading training dataset....")
            self.train_data = self.dataset_type(root=self.data_dir,
                                                split='train',
                                                inp_transform=train_inp_transform,
                                                target_transform=train_tgt_transform,
                                                randflip=self.randflip,
                                                fold=self.fold)
            self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size,
                                num_workers=self.num_workers, pin_memory=True, drop_last=True,
                                persistent_workers=True)

        print(f"** Loading validation dataset....")
        self.val_data = self.dataset_type(root=self.data_dir,
                                          split='val',
                                          inp_transform=None,
                                          target_transform=None,
                                          fold=self.fold)
        self.val_loader = DataLoader(self.val_data, batch_size=1,
                            num_workers=self.num_workers, pin_memory=True, drop_last=False,
                            persistent_workers=True)

        # print(f"** Loading testing dataset....")
        # self.test_data = self.dataset_type(root=self.data_dir,
        #                                   split='test',
        #                                   inp_transform=None,
        #                                   target_transform=None,
        #                                   fold=self.fold)
        # self.test_loader = DataLoader(self.test_data, batch_size=1,
        #                     num_workers=self.num_workers, pin_memory=True, drop_last=False,
        #                     persistent_workers=True)

    def train_dataloader(self):
        '''
        without persistent_workers=True, the dataloader workers are
        killed and created after every epoch. For small datasets, this
        is a problem.
        '''
        # return DataLoader(self.train_data, batch_size=self.batch_size,
        #                     num_workers=self.num_workers, pin_memory=True, drop_last=True,
        #                     persistent_workers=True)
        return self.train_loader

    def val_dataloader(self):
        # return DataLoader(self.val_data, batch_size=self.batch_size,
        #                     num_workers=self.num_workers, pin_memory=True, drop_last=False,
        #                     persistent_workers=True)
        return self.val_loader

    def test_dataloader(self):
        # return DataLoader(self.test_data, batch_size=self.batch_size,
        #                     num_workers=self.num_workers, pin_memory=True, drop_last=False,
        #                     persistent_workers=True)

        return self.test_loader

