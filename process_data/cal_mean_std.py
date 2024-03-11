import numpy as np
import os
import scipy.io as scio
import torch

# 读取mat文件名列表
matfile = []
root = '~/workspace/zx/HADAR_database/'
for i in list(range(1,11)):
    filename = os.listdir(os.path.join(root, f'Scene{i}', 'HeatCubes'))
# print(filename)
# print(len(filename))
    for file in filename:
        if file[-3:] == 'mat':
            matfile.append(os.path.join(root, f'Scene{i}', 'HeatCubes') + '/' + file)
print(matfile)
print(len(matfile))

# 读取数据
data = []
for mat in matfile:
    sub_data = scio.loadmat(mat)["S"]
    data.append(sub_data)
data = np.array(data)
print(data.shape)


data_ = np.transpose(data, [3, 0, 1, 2])
print(data_.shape)

mean = []
std = []
for i in range(data_.shape[0]):
    temp = []
    for j in range(data_.shape[1]):
        for k in range(data_.shape[2]):
            for l in range(data_.shape[3]):
                temp.append(data_[i][j][k][l])
    # sum.append(temp)
    std.append(np.std(temp, dtype=np.float64))
    mean.append(np.mean(temp, dtype=np.float64))
    print(np.std(temp), np.mean(temp))
print(std,'\n')
print(mean)

'''mean = np.mean(data, axis=(0, 1, 2),dtype=np.float64)  # 计算均值，axis指定了要沿着哪些轴计算均值
std = np.std(data, axis=(0, 1, 2),dtype=np.float64)    # 计算标准差

print("均值", mean.shape, '\n', mean)
print("标准差:", std.shape, '\n', std)'''

'''
data_tensor = torch.tensor(data)

# 将数据平均分配到两个GPU上
device1 = torch.device("cuda:0")  # 第一个GPU
device2 = torch.device("cuda:1")  # 第二个GPU

data_gpu1 = data_tensor[:50].to(device1)  # 将前50个样本放在第一个GPU上
data_gpu2 = data_tensor[50:].to(device2)  # 将后50个样本放在第二个GPU上

# 计算每个GPU上最后一个通道维度的标准差
# std_gpu1 = torch.std(data_gpu1, dim=(0, 1, 2), dtype=torch.float64)
# std_gpu2 = torch.std(data_gpu2, dim=(0, 1, 2), dtype=torch.float64)
std_gpu1 = torch.std(data_gpu1, dim=(0, 1, 2), unbiased=False, keepdim=False)
std_gpu2 = torch.std(data_gpu2, dim=(0, 1, 2), unbiased=False, keepdim=False)

print('gpu1 \n', std_gpu1)
print(std_gpu2)

# 将std_gpu2移动到device1上
std_gpu2 = std_gpu2.to(device1)

# 合并两个GPU上的标准差
std = torch.cat((std_gpu1.unsqueeze(0), std_gpu2.unsqueeze(0)), dim=0)      # cat是连接起来，不能直接合并，但是直接算内存又不够
# std = torch.cat((std_gpu1.unsqueeze(0), std_gpu2.unsqueeze(0)), dim=0).cpu().numpy()
print("标准差:", std.shape, '\n', std)

# # 合并两个GPU上的标准差
# all_std = torch.cat((std_gpu1.unsqueeze(0), std_gpu2.unsqueeze(0)), dim=0)

# # 计算合并后的标准差
# final_std = torch.std(all_std, unbiased=False)
# print("标准差:", final_std.shape, '\n', final_std)'''



'''
结果
tensor([[0.0128, 0.0128, 0.0129, 0.0127, 0.0128, 0.0127, 0.0129, 0.0132, 0.0127,
         0.0123, 0.0122, 0.0121, 0.0121, 0.0124, 0.0121, 0.0120, 0.0121, 0.0118,
         0.0118, 0.0117, 0.0117, 0.0115, 0.0115, 0.0114, 0.0113, 0.0113, 0.0111,
         0.0111, 0.0111, 0.0111, 0.0111, 0.0110, 0.0109, 0.0109, 0.0108, 0.0108,
         0.0097, 0.0094, 0.0093, 0.0091, 0.0089, 0.0090, 0.0088, 0.0086, 0.0086,
         0.0085, 0.0084, 0.0082, 0.0082, 0.0089, 0.0079, 0.0079, 0.0078, 0.0077],
        [0.0125, 0.0125, 0.0125, 0.0126, 0.0126, 0.0126, 0.0126, 0.0126, 0.0125,
         0.0123, 0.0122, 0.0121, 0.0123, 0.0123, 0.0122, 0.0120, 0.0120, 0.0119,
         0.0120, 0.0119, 0.0118, 0.0117, 0.0117, 0.0117, 0.0114, 0.0114, 0.0113,
         0.0116, 0.0115, 0.0113, 0.0111, 0.0109, 0.0108, 0.0108, 0.0106, 0.0105,
         0.0103, 0.0101, 0.0100, 0.0098, 0.0096, 0.0095, 0.0094, 0.0092, 0.0090,
         0.0089, 0.0087, 0.0086, 0.0084, 0.0083, 0.0081, 0.0080, 0.0078, 0.0077]],
       device='cuda:0')
'''