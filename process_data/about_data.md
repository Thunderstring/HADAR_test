## Dataset
The directory names of the downloaded data are renamed to Scene1, Scene2, ..., Scene10.

`matName_FullDatabase.mat`: the materials in the dataset, 30 kinds in total. 

sky asphalt tilePavement cinderblock brick wall marble stone soil al weatheredMetal brass al2o3 oxidizedSteel iron

carpaint plasticPaint yellowSpray carwindow crystalGlass tire cloths card tree grass flower water human bark concrete

## HeatCubes
Scene1-10

L/R_0001_heatcube.mat 文件  原始数据 观测到的HeatCubes  双目，分左右，一个五帧，左右共十帧

Scene1-10变量名为S  (1080,1920,54)      

Scene11  单目，四帧  变量名为HSI (260,1500,49)

L/R_0001_heatcube..npy  由.mat转换而来，为了读取更快速

## S_EnvObj_L/R_0001.npy    mean of the multispectral heatcubes, S_beta

用于记录显著物体，此处是两个，通过L_0001_heatcube.mat  处理得到（见preprecess_data.ipynb）
Scene1-10: (1, 54, 2, 1)   Scene11: (1, 49, 2, 1)
 
For scene 1-10, generate S_EnvObj_L/R_000X.npy from the heatcubes. 

Heatcube: 1080x1920x54 ---> 1x54x1080x1920 --AvgPooling--> 1x54x2x1. 

For scene 11, loaded from mat. 1x49x2x1. 

The heatcube is divided into top (cloudy sky) and bottom (ground) halfs and then calculate their mean respectively. 

See the Equation S48 in Nature paper's SI. 

## eMap
`/groundtruth/eMap/eMap_L(R)_000X.mat`: index of the materials in eList, values in 1-len(eList). It can be seen as the local scene index. 

eList preprocessing change the local index to global index. The results are save as 
`/groundtruth/eMap/new_eMap_L(R)_000X.npy`


## tMap
GroundTruth fot T

Scene1-10: (1080,1920)   Scene11: (260 *1500)