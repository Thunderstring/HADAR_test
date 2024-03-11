HADAR database is an LWIR (long-wave infrared) hyperspectral and stereo database to develop algorithms for thermal navigation. It can be used to develop algorithms of TeX decomposition to generate TeX vision, or to develop algorithms about object detection, semantic/scene segmentation, optical/scene flow, stereo depth, etc. based on TeX vision instead of traditional RGB or thermal vision.

The current HADAR database consists of 11 datasets, with scenes ranging from Crowded street, Highway, to Suburb, Countryside, to Indoor, and to Forest and Desert, covering most common road conditions that HADAR may find applications in. The first 10 scenes are synthetic datasets mimicing self driving situations, with two HADAR sensors, left (L) and right (R), either mounted at the positions of headlights, or on the top of the automated vehicles, or on robot helpers. Each scene has two views from two stereo sensors. Each view has 5 frames. Each frame is a Height*Width*Channel = 1080*1920 * 54 heat cube. These synthetic datasets are clean pass without motion or focus blur. The Street scene has a 100-frame long animation, of weaker surface normal variations (weaker roughness on the road), with settings particularly specified. The 11th dataset is a real-world experimental scene with only one view and 4 frames. Each frame is a Height*Width*Channel = 260*1500*49 heat cube. For the Street, Suburb, and RockyTerrain scenes, RGB vision images are also generated as a baseline for the purpose of depth estimation. In monocular stereo vision, TeX, RGB and IR images are also provided for the real-world experimental scene.

--- Dr. Fanglin Bao. Sep. 29, 2022. (baof:at:purdue:dot:edu)

====================================================================================================
////////////////////////////// Technical details about the database ////////////////////////////////
====================================================================================================

matName_FullDatabase.mat, HueLib_FullDatabase.mat, matLib_FullDatabase.mat give the names, Hues, and spectral emissivities, respectively, for all materials in the full database including the experimental scene.

All synthetic datasets share the same wavenumbers, 720:10:1250, in unit of 1/cm, given in wavenumber_syntheticscenes.mat. The experimental dataset has wavenumbers 760:10:1240 given in wavenumber_experimentalscene.mat, corresponding to the 5th:1:53rd bands of the synthetic scenes.

matLib_FullDatabase has 30 materials but 49 bands, while matLib_SyntheticScenes has 28 materials but 54 bands. The latter gives the spectral emissivities for all materials in synthetic scenes.

For the experimental scene, the ground truth matLib is estimated by TES (temperature-emissivity separation) followed by K-means clustering. The ground truth tMap, eMap, vMap, and TeX vision are estimated by our proposed TeX-SGD (semi-global decomposition).

For synthetic scenes, Ray depth (max. bounces of light) = 4.

Sampling per pixel = 2000.

All surfaces are completely diffusive.

No gamma correction or smoothing applied to data.

For all scenes, Ground truth depth is in unit of meters. tMap is in unit of celsius degree. eList.mat in each scene gives the indices of the used subset of the full material library. eMap of each scene is the index in the subset material library.

The provided vMap, resMap, and TeX vision in the ground truth folder is a sample case. Environmental objects can be re-defined by users, depending on the scene, and vMap/resMap/TeX will vary accordingly. HueLib_full can also be re-defined by users to customize the material colors in TeX vision.

The distance between left and right sensors (stereo baseline) b = 0.05 meters for the Indoor scene; b = 0.4 meters for all other synthetic scenes.

Stereo sensors are looking forward in parallel. One exception is the last frame of the Forest scene, where Left camera Z rotation is 328.652d while Right camera is 325.523d.

Focal length and Field of view are given in HADAR_sensor_config.xlsx

==========================================================================================
////////////////////////////// matlab code for TeX vision ////////////////////////////////
==========================================================================================

TeX is a matlab class of functions for TeX vision. It provides a non-machine-learning algorithm, the TeX-SGD (Semi-Global Decomposition), to decompose TeX out of the heatcube, an implementation of texture distillation (distillX), and the codes to visualize the TeX vision. TeX decomposition can also be done by the TeX-Net and gets visualized here. Sample application codes have been given in mainTeX.m. The blackbody radiation can be called through TeX.BBp. The equations of the blackbody radiation have also been provided separately in blackbodyradiation.m.

==========================================================================================
///////////////////////////////////// TeX-Net code ///////////////////////////////////////
==========================================================================================

The TeX-Net folder contains the TeX-Net model and pre-trained weights.

==========================================================================================
////////////////////////////////////// TeX video /////////////////////////////////////////
==========================================================================================

Demonstrations of TeX vision in a real-world off-road scene as well as a synthetic on-road scene at night are provided, in comparison with thermal vision.
