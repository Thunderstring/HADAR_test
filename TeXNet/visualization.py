import torch
import torch.nn.functional as F
import numpy as np
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import os
import sys
from tqdm import trange
import scipy.special as scsp
from skimage import img_as_float
from skimage import exposure
import torchmetrics

# Give data location (folder name) in DATA_DIR and
# visualization results in OUT_DIR


DATA_DIR = '/mnt/Disk/zx/HADAR/test3_integral/6-channels_val_new'
OUT_DIR = DATA_DIR+'/visualized'

num_files = 40
visualize_flag = True
# visualize_flag = False

max_n_class = 30
# T_max = 70

e_accuracy = []
T_err_mean = []
v_kldiv_mean = []
v_error_mean = []
S_error = []



if visualize_flag:
    os.makedirs(OUT_DIR, exist_ok=True)

def get_X_from_V(S, v):
    C, H, W = S.shape
    S1 = (np.mean(S[:, :H//2]))
    # S2 = (np.mean(S[:, H//2:]))     # ?
    S2 = (np.mean(S[:, H//2:]))     
    # print(v[0])

    X = v[0]*S1 + v[1]*S2

    return X

def visualize_TeX(TeX, fname, max_vals, kind='pred'):       # 这个可视化的还是TeX
    T_max = max_vals[0]
    S_max = max_vals[1]
    assert len(TeX.shape) == 3
    assert kind in ['pred', 'gt', 'residue']
    if kind == 'pred':
        kind = 'Prediction'
    elif kind == 'gt':
        kind = 'GT'
    else:
        kind = "Residue"

    # Need to normalize each value of TeX to [0, 1]
    TeX[..., 0] /= max_n_class # Divide e-map by the number of classes
    TeX[..., 1] /= T_max # Divide the T-map by the maximum temperature.
    TeX[..., 2] /= S_max # divide by the maximum of np.mean(S_half) among each half.

    TeX_ = mpl.colors.hsv_to_rgb(TeX)/np.amax(mpl.colors.hsv_to_rgb(TeX))
    TeX_ = ((TeX_ - np.min(TeX_))/(np.max(TeX_) - np.min(TeX_)*255.).astype(np.uint8))
    plt.imshow(TeX_)
    plt.title('TeX '+ kind)
    plt.axis('off')
    # plt.colorbar()
    plt.clim(0,1)
    plt.savefig(os.path.join(OUT_DIR, fname))
    plt.close()

    return

def visualize_residue(i, j, res, fname, kind = 'S_residue'):
    # Need to normalize each value of TeX to [0, 1]

    img = np.squeeze(np.linalg.norm(res, axis=0))
    if kind == 'S_residue':
        np.save(OUT_DIR+'/'+f'residue_{i+j}_no_rescale.npy', img)
    elif kind == 'S_True':
        np.save(OUT_DIR+'/'+f'S_True_{i+j}_no_rescale.npy', img)
    elif kind == 'S_pred':
        np.save(OUT_DIR+'/'+f'S_pred_{i+j}_no_rescale.npy', img)
    img = img_as_float(np.squeeze(np.linalg.norm(res, axis=0)))/255.0
    # print(np.max(img))

    img_adapteq = exposure.equalize_adapthist(img_as_float(img), clip_limit=0.2)

    # np.save(f'residue_{i}_no_rescale.npy', img)
    plt.imshow(img_adapteq)
    plt.title(kind)
    plt.axis('off')
    plt.colorbar()
    plt.clim(0,1)
    plt.savefig(os.path.join(OUT_DIR, fname))
    plt.close()
    return

def visualize_T(data, fname, error=False, kind='pred'):
    assert len(data.shape) == 2
    assert kind in ['pred', 'gt']
    kind = 'Prediction' if kind == 'pred' else 'GT'

    mu = 15.997467357494212
    std = 8.544861474951992

    # mu = np.load('/home/sureshbs/Desktop/TeXNet/Dataset/HADAR_database/Scene14/GroundTruth/tMap/T_mu.npy')
    # std = np.load('/home/sureshbs/Desktop/TeXNet/Dataset/HADAR_database/Scene14/GroundTruth/tMap/T_std.npy')

    # denormalize temperature data, if we are not plotting error
    if not error:
        data = data*std + mu
    else:
        data = data*std


    plt.imshow(data, cmap='turbo', vmin=0, vmax=70)
    if not error:
        plt.title('Temperature '+kind)
    else:
        plt.title('L1 error in temperature prediction')
    plt.axis('off')
    plt.colorbar(shrink=0.5)
    plt.savefig(os.path.join(OUT_DIR, fname))
    plt.close()

    return

def visualize_m_CE(data, fname):
    assert len(data.shape) == 2

    # Even though CE is unbounded, it will not be higher than 4 unless
    # it is a very wrong prediction, at which point we are not concerned
    # about how bad it is.
    plt.imshow(data, cmap='turbo', vmin=0, vmax=2)
    plt.title('Cross entropy error in e-map prediction')
    plt.axis('off')
    plt.colorbar()
    plt.savefig(os.path.join(OUT_DIR, fname))
    plt.close()
    return

def visualize_m(data, fname, kind='pred'):
    assert kind in ['pred', 'gt']
    kind = 'Prediction' if kind == 'pred' else 'GT'

    # hue = np.array([[0.5569],
    #                [0.5529],
    #                [0.3098],
    #                [0.8588],
    #                [0.4196],
    #                [0.2   ],
    #                [0.902 ],
    #                [0.5833],
    #                [0.1386],
    #                [0.8   ],
    #                [0.78  ],
    #                [0.76  ],
    #                [0.74  ],
    #                [0.72  ],
    #                [0.7   ],
    #                [0.95  ],
    #                [0.93  ],
    #                [0.91  ],
    #                [0.451 ],
    #                [0.41  ],
    #                [0.6627],
    #                [0.1   ],
    #                [0.1586],
    #                [0.04  ],
    #                [0.2641],
    #                [0.97  ],
    #                [0.5693],
    #                [0.    ],
    #                ])
    hue = np.array([[0.5098],           # 30种材料对应的hue
                    [0.6157],
                    [0.8784],
                    [0.0431],
                    [0.0745],
                    [0.1059],
                    [0.3451],
                    [0.0196],
                    [0.1176],
                    [0.5059],
                    [0.0588],
                    [0.0941],
                    [0.5961],
                    [0.5255],
                    [0.0784],
                    [0.0039],
                    [0.5373],
                    [0.1294],
                    [0.4510],
                    [0.1529],
                    [0.1843],
                    [0.4784],
                    [0.0706],
                    [0.0392],
                    [0.3137],
                    [0.8706],
                    [0.5882],
                    [0.9373],
                    [0.0392],
                    [0.1961]])

    sat = np.ones_like(hue)*0.7     # satuation 饱和度
    val = np.ones_like(hue)         # value     亮度

    hsv = np.concatenate((hue, sat, val), 1)        # 映射
    rgb = mpl.colors.hsv_to_rgb(hsv)        # hsv转为rgb用于可视化
    rgb = np.concatenate((rgb, np.ones((max_n_class, 1))), 1)

    newcmap = ListedColormap(rgb)

    mycmap = plt.get_cmap('gist_rainbow', max_n_class)

    plt.imshow(data, cmap=mycmap, vmin=0, vmax=max_n_class-1)
    plt.title('Material map '+kind)
    plt.axis('off')
    cbar = plt.colorbar()
    cbar.set_ticks([])
    plt.savefig(os.path.join(OUT_DIR, fname))
    plt.close()

    return

def visualize_m_error(data, fname):
    mycmap = plt.get_cmap('gray', 2)
    plt.imshow(data, cmap=mycmap)
    plt.title('Error in material map')
    plt.axis('off')
    cbar = plt.colorbar()
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['correct', 'wrong'])
    plt.savefig(os.path.join(OUT_DIR, fname))
    plt.close()
    return

def save_m_file(data, fname):
    np.save(os.path.join(OUT_DIR, fname), data)


def visualize_v(data, fname, kind='pred'):
    assert kind in ['pred', 'gt', 'l1error', 'kldiv']

    vmax = 1.
    if kind == 'gt':
        kind = 'GT'
    elif kind == 'pred':
        kind = 'Prediction'
    elif kind == 'l1error':
        kind = 'L1 error'
    elif kind == 'kldiv':
        kind = 'KL-div'
        vmax = 4

    if kind != 'KL-div':
        v1 = np.squeeze(data[0, ...])
        v2 = np.squeeze(data[1, ...])
        #v3 = np.squeeze(data[2, ...])
        #v4 = np.squeeze(data[3, ...])

        fig, ax = plt.subplots(1, 2)
        im = ax[0].imshow(v1, vmin=0., vmax=vmax, cmap='turbo')
        ax[0].axis('off')
        ax[0].set_title('v1')
        # plt.colorbar(im, ax=ax[0],shrink=0.5)
        plt.colorbar(im, ax=ax[0])

        im = ax[1].imshow(v2, vmin=0., vmax=vmax, cmap='turbo')
        ax[1].axis('off')
        ax[1].set_title('v2')
        # plt.colorbar(im, ax=ax[1],shrink=0.5)
        plt.colorbar(im, ax=ax[1])

        #im = ax[1, 0].imshow(v3, vmin=0., vmax=vmax, cmap='turbo')
        #ax[1, 0].axis('off')
        #ax[1, 0].set_title('v3')
        #plt.colorbar(im, ax=ax[1, 0])

        #im = ax[1, 1].imshow(v4, vmin=0., vmax=vmax, cmap='turbo')
        #ax[1, 1].axis('off')
        #ax[1, 1].set_title('v4')
        #plt.colorbar(im, ax=ax[1, 1])

        plt.suptitle('v-map '+kind)

        fig.savefig(os.path.join(OUT_DIR, fname))
        plt.clf()
        plt.close()
    else:
        fig, ax = plt.subplots()
        im = ax.imshow(data, vmin=0., vmax=vmax, cmap='turbo')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
        plt.suptitle('v-map '+kind)

        fig.savefig(os.path.join(OUT_DIR, fname))
        plt.clf()
        plt.close()

    return

# Calculate the metrics for each large and small samples
large_miou = []
small_miou = []
syn_miou = []
exp_miou = []

print("folder: ", DATA_DIR)

for j in trange(num_files):     # 42 ?    why 42?   Scene1-10 的L、R两张，L、R的crop，Scene11 一张原图和crop用于val
    # 此.pt文件保存的是t,e,v,pred,S_pred,img(true)数据（裁剪后）
    T_file = torch.load(os.path.join(DATA_DIR, f'val_T_{j}.pt'), map_location='cpu')    # [1, 1, 256, 256]  e,T,v_ground truth
    e_file = torch.load(os.path.join(DATA_DIR, f'val_e_{j}.pt'), map_location='cpu')    # [1, 256, 256]
    v_file = torch.load(os.path.join(DATA_DIR, f'val_v_{j}.pt'), map_location='cpu')    # [1, 2, 256, 256]
    pred_file = torch.load(os.path.join(DATA_DIR, f'val_pred_{j}.pt'), map_location='cpu')  # [1, 33, 256, 256]   30(materials) + 1(T) + 2(V) = 33
    S_pred_file = torch.load(os.path.join(DATA_DIR, f'val_S_pred_{j}.pt'), map_location='cpu')  # [1, 49, 256, 256]
    S_true_file = torch.load(os.path.join(DATA_DIR, f'val_S_true_{j}.pt'), map_location='cpu')  # [1, 49, 256, 256]
    
    assert T_file.size(0) == e_file.size(0) == v_file.size(0) == pred_file.size(0)  # 确保第一个维度（batch 维度）的大小相等
        # 这batch怎么是1呢？不是只有验证时batch是1吗
    n = T_file.size(0)

    nclass = max_n_class

    for i in range(n):
        pred = pred_file[i].squeeze()
        e = e_file[i].squeeze()         # e,T,v_ground truth

        T = T_file[i].squeeze().numpy()

        v = v_file[i].squeeze().numpy()

        c = pred.size(0)
        e_pred = pred[:nclass]      # pred里面取出前30个通道，是材料种类
        T_pred = None               # 初始化T和v
        v_pred = None
        if c == nclass+1:       # 30(materials) + 1(T)
            T_pred = pred[nclass]
        elif c == nclass+4:
            v_pred = pred[nclass:]
        elif c == nclass+3:     # 30(materials) + 1(T) + 2(V)
            T_pred = pred[nclass]   # pred里面取出第31个通道，温度T
            v_pred = pred[nclass+1:]    # pred里面取出最后两个通道，热照明因子v

            # e_pred = F.softmax(e_pred, 0).squeeze()

        if nclass == max_n_class:
            # e_pred_ = torch.argmax(e_pred, 0, keepdim=False)
            e_pred_ = torch.argmax(e_pred.unsqueeze(0), 1, keepdim=False)   # 计算每个元素上，材料最大可能预测类别
            e_ = e.unsqueeze(0)     # 材料GT

            miou = torchmetrics.classification.MulticlassJaccardIndex(num_classes=nclass)
            val = miou(e_pred_, e_).item()      # 预测结果 e_pred_ 和真实标签 e_ 之间的miou
            # 数值越接近 1 表示模型预测结果与真实标签的重叠程度越高，即性能越好
            # append to the right type of metric
            if e.shape[0] == 256:
                small_miou.append(val)
            elif e.shape[0] == 1080:
                large_miou.append(val)
                syn_miou.append(val)
            elif e.shape[0] == 260:
                large_miou.append(val)
                exp_miou.append(val)

            del miou

            e_ce_error = F.cross_entropy(e_pred.unsqueeze(0), e.unsqueeze(0), reduction='none').squeeze()   # 材料预测值与真实值的交叉熵
            e = e.numpy()
            e_pred = torch.argmax(e_pred, 0).squeeze().numpy()
            e_error = (e_pred != e).astype(int)     

            if visualize_flag :
                visualize_m(e_pred, fname=f'emap_pred_{i+j*n}.png', kind='pred')    # emap的预测
                visualize_m(e, fname=f'emap_GT_{i+j*n}.png', kind='gt')             # emap的GT
                # print("e error", j, i, np.mean(e_error.astype(float)))
                visualize_m_error(e_error, fname=f'emap_error_{i+j*n}.png')         # 每个像素上材料种类的预测误差，正确为0，错误为1
                visualize_m_CE(e_ce_error, fname=f'emap_CE_error_{i+j*n}.png')      # emap的cross entropy error
                save_m_file(e_pred, fname=f'm_pred_{i+j*n}.npy')        

            e_accuracy.append(np.mean(e_error)) 

        if T_pred is not None:
            T_pred = T_pred.squeeze().numpy()
            T_error = np.abs(T-T_pred)

            if visualize_flag:
                visualize_T(T_pred, fname=f'Tmap_pred_{i+j*n}.png', error=False, kind='pred')   # Tmap预测值
                visualize_T(T, fname=f'Tmap_GT_{i+j*n}.png', error=False, kind='gt')            # Tmap的ground truth
                save_m_file(T_pred, fname=f'T_pred_{i+j*n}.npy')            
                visualize_T(T_error, fname=f'Tmap_error_{i+j*n}.png', error=True)               # Tmap的误差

            # T_err_mean.append(np.mean(np.abs(T_error)/np.abs(T)))
            T_err_mean.append(np.mean(np.abs(T_error)))   # 用L1 error

        if v_pred is not None:
            v_pred = v_pred.squeeze()
            v = v.squeeze()
            v_pred = F.softmax(v_pred, 0).numpy()   # softmax将张量的每个元素转换为0到1之间的概率值，同时确保所有元素的总和为1
            
            v_error = np.abs(v-v_pred)
            v_kldiv = scsp.rel_entr(v, v_pred).sum(0)

            if visualize_flag:
                save_m_file(v_pred, fname=f'v_pred_{i+j*n}.npy')    
                visualize_v(v_pred, fname=f'vmap_pred_{i+j*n}.png', kind='pred')        # vmap预测值
                visualize_v(v, fname=f'vmap_GT_{i+j*n}.png', kind='gt')                 # vmap的ground truth
                visualize_v(v_error, fname=f'vmap_error_{i+j*n}.png', kind='l1error')   # vmap的误差
                visualize_v(v_kldiv, fname=f'vmap_KLdiv_{i+j*n}.png', kind='kldiv')     # vmap的KL散度（相对熵）
            v_kldiv_mean.append(np.mean(v_kldiv)) 
            v_error_mean.append(np.mean(v_error))

        # Load S_pred and S_true here.          这个S_true有问题，与S_pred相差过多，且S_pred是正常的，可以正常显示图像
        S_pred = S_pred_file[i].squeeze().numpy()   
        S_true = S_true_file[i].squeeze().numpy()
        # 从thermal lighting factors V中得到texture X
        X_pred = get_X_from_V(S_pred, v_pred)   
        X_true = get_X_from_V(S_true, v)

        # print(np.shape(np.dstack((e_pred, T_pred, X_pred))))

        # TeX_pred = np.concatenate((e_pred, T_pred, X_pred), 2)
        # TeX_true = np.concatenate((e, T, X_true), 2)
        # 全体起立！ TeX成像辣！
        TeX_pred = np.dstack((e_pred, T_pred, X_pred))  # 对数组进行堆叠,TeX成像
        TeX_true = np.dstack((e, T, X_true))

        # Todo
        S_res = (np.log(np.abs(S_true-S_pred)))     # 计算预测值与真实值的残差    归一化后的img（很大）和没归一化的S_pred（很小），相减后还是img，所以可能错误在这      
        
        # TeX Vision是根据S_res算出来的！！！把S_res作为Xmap 我真的是服了
        # TeXnet的visualization需要归一化后的S_pred_，因为输入的img也是归一化后的
        # TeX Vision需要归一化后的img（很大）和没归一化的S_pred（很小），相减后还是img，所以可能错误在这，TeX Vision明明是直接用归一化后的img去做了TeX Vision
        # 可是按他这么来就是有纹理，如果只用S_pred的话，第一个场景的人就没纹理
        # 可是27channels的用S_pred，第一个场景的人就有纹理 ，49channels的就没有，真玄学
        # 这得好好看看，不知道改得对不对，按理说归一化就应该都归一化

        # 若用归一化后的S_true-S_pred相减，也可以得到TeXvision，只不过很多噪点，看来确实是相减的resmap有纹理，看一下论文咋说的，把原理搞懂
        # 看一下论文吧，把resmap搞懂

        # T_max = np.max(np.maximum(T, T_pred))
        T_max1 = np.max(T_pred)
        T_max2 = np.max(T)
        C, H, W = S_pred.shape
        S_max1 = np.maximum(np.mean(S_pred[:, :H//2]), np.mean(S_pred[:, H//2:]))
        S_max2 = np.maximum(np.mean(S_true[:, :H//2]), np.mean(S_true[:, H//2:]))

        # S_error.append(np.mean(np.abs(S_true-S_pred)/np.abs(S_true)))
        S_error.append(np.mean(np.abs(S_true-S_pred)))      # L1 误差

        # S_max = np.maximum(S_max1, S_max2)
        # S_max = np.maximum(S_max1, S_max1)
        # S_max = np.maximum(S_max2, S_max2)

        if visualize_flag:
            # visualize_TeX(TeX_pred, fname=f'TeX_pred_{i+j*n}.png', max_vals=[T_max, S_max], kind='pred')
            # visualize_TeX(TeX_true, fname=f'TeX_GT_{i+j*n}.png', max_vals=[T_max, S_max], kind='gt')
            visualize_TeX(TeX_pred, fname=f'TeX_pred_{i+j*n}.png', max_vals=[T_max1, S_max1], kind='pred')
            visualize_TeX(TeX_true, fname=f'TeX_GT_{i+j*n}.png', max_vals=[T_max2, S_max2], kind='gt')
            visualize_residue(i, j, S_true, fname=f'S_true_{i+j*n}.png', kind = 'S_True')
            visualize_residue(i, j, S_pred, fname=f'S_pred_{i+j*n}.png', kind = 'S_pred')
            visualize_residue(i, j, S_res, fname=f'S_residue_{i+j*n}.png',kind = 'S_residue')  # heatcube预测值与真实值的残差


print("Average large mIoU", np.mean(large_miou))        # iou在0-1之间，越接近1表示预测结果越好
print("Average small mIoU", np.mean(small_miou))
print("Average synthetic mIoU", np.mean(syn_miou))
print("Average experimental mIoU", np.mean(exp_miou))
# accuracy
print("e_error", np.mean(e_accuracy))
print("T_error", np.mean(T_err_mean))
print("v_kldiv", np.mean(v_kldiv_mean))
print("v_error", np.mean(v_error_mean))
print("S_error", np.mean(S_error))


with open(os.path.join(DATA_DIR, 'results.txt'), 'a') as metrics:
    metrics.write("\n")
    # metrics.write("Average large mIoU: " + f"{np.mean(large_miou)} \n")
    # metrics.write("Average small mIoU: " + f"{np.mean(small_miou)} \n")
    # metrics.write("Average synthetic mIoU: " + f"{np.mean(syn_miou)} \n")
    # metrics.write("Average experimental mIoU: " + f"{np.mean(exp_miou)} \n")
    metrics.write("e_error: " + f"{np.mean(e_accuracy)} \n")
    metrics.write("T_error: " + f"{np.mean(T_err_mean)} \n")
    metrics.write("v_kldiv: " + f"{np.mean(v_kldiv_mean)} \n")
    metrics.write("v_error: " + f"{np.mean(v_error_mean)} \n")
    metrics.write("S_error: " + f"{np.mean(S_error)} \n")
