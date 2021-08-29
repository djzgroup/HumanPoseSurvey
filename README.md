# HumanPoseSurvey
## Deep learning methods for 3D human pose estimation under different supervision manners: A survey
3D human pose estimation is an essential technique for many computer vision applications such as video surveillance, human-computer interaction, digital entertainment, etc. Meanwhile, it is also a challenging task due to the inherent ambiguity and occlusion problems. Lately, deep learning method for 3D human pose estimation has attracted increasing attention due to the success of learning based 2D human pose estimation. Against this backdrop, numerous methods have being proposed to address different problems in this area. Generally, these methods consist of the body modeling, learning based pose estimation and regularization for refinement. This paper provides an extensive literature survey of recent literatures about Deep learning methods for 3D human pose estimation, covering the aspects of human body model, 3D single-person pose estimation, 3D multi-person human pose estimation and regularization. Moreover, as deep learning methods can be categorized according to supervision manners which are based on quite different philosophies and are applicable for diverse scenarios, this paper provides a classification for both 3D single-person and multi-person pose estimation methods based on their supervision manners, i.e. unsupervised methods, fully-supervised methods, weakly-supervised methods, semi-supervised methods, and self-supervised methods. At last, this paper also enlists the contemporary and widely used datasets, compares performances of reviewed methods and discusses promising research directions.

**Keywords:** 3D human pose estimation; deep learning; unsupervised; fully-supervised; weakly-supervised; semi-supervised

## Single-person and Multi-person

- ***Single-person 3D pose estimation*** falls into two categories: two-stage and One-stage methods. 
  - *Two-stage methods* involve two steps, first, 2D joint locations are obtained by 2D keypoints detection models, then 2D keypoints are lifted to 3D keypoints by deep learning methods. 
  - *One-stage methods* mean regressing 3D joint locations directly from a RGB image. These methods require many training data with 3D annotations, but manual annotation is costly and demanding.
- ***Multi-person 3D pose estimation*** is divided into two categories: top-down and bottom-up methods. 
  - *Top-down methods* first detect the human candidates and then apply single-person pose estimation for each of them. 
  - *Bottom-up methods* first detect all keypoints followed by grouping them into different people. 

## Input form

- ***RGB image-based methods*** take static images as input, only taking spatial context into account, which differs from video-based methods.
- ***Video-based methods*** meet more challenges than image-based methods, such as temporal information processing, correspondence between spatial information and temporal information and motion changes in different frames, etc.


## Supervision form
- ***Unsupervised methods*** do not require any multi-view image data, 3D skeletons, correspondences between 2D-3D points, or use previously learned 3D priors during training. Self-supervised methods which can also solve the issue, deficiency of 3D data, have become popular in recent years. *Self-supervised methods* is a form of unsupervised learning where the data provides the supervision.
- ***Fully-supervised methods*** rely on large training sets annotated with ground-truth 3D positions coming from multi-view motion capture systems.
- ***Weakly-supervised methods*** access multiple cues for weak supervision, such as, a) paired 2D ground-truth, b) unpaired 3D ground-truth (3D pose without the corresponding image), c) multi-view image pair, d) camera parameters in a multi-view setup, etc.
- ***Semi-supervised methods*** use part of annotated data (e.g. 10 percent of 3D labels), which means labeled training data is scarce.


## The taxonomy of deep learning methods for 3D human pose estimation
Both *single-person 3D pose estimation* and *multi-person 3D pose estimation* combined with different supervision forms could derive various branches as described in the figure below.

<img src="https://github.com/djzgroup/HumanPoseSurvey/blob/main/taxonomy.png" width="550">

It is an unbalanced tree describing deep learning based 3D human pose estimation. Multi-person 3D pose estimation has received less interest compared to single-person 3D pose estimation. Also, video-based 3D pose estimation is less studied than image-based 3D pose estimation. Another interesting sight is that fully-supervised methods are presented in each sub-category, which may indicate that fully-supervised methods are helped to investigate a research area at the beginning. 


## Summary on datasets
We present the state-of-the-art results on several datasets, such as Human3.6m, MPI-INF-3DHP, MuPoTS-3D, Shelf, and Campus datasets.

### Summary of the state-of-the-arts methods on Human3.6M dataset.

| **Title**                                                    | **Year** | **Supervision**   | **Type**   |                           **URL**                            |
| ------------------------------------------------------------ | -------- | ----------------- | ---------- | :----------------------------------------------------------: |
| [3d human pose estimation from monocular images with deep convolutional neural  network](https://link.springer.com/chapter/10.1007/978-3-319-16808-1_23) | 2014     | fully-supervised  | monocular  |                              -                               |
| [Sparseness  Meets Deepness: 3D Human Pose Estimation from Monocular Video](https://openaccess.thecvf.com/content_cvpr_2016/html/Zhou_Sparseness_Meets_Deepness_CVPR_2016_paper.html) | 2016     | weakly-supervised | monocular  |      [code](http://cis.upenn.edu/˜xiaowz/monocap.html)       |
| [Structured  Prediction of 3D Human Pose with Deep Neural Networks](http://www.bmva.org/bmvc/2016/papers/paper130/paper130.pdf) | 2016     | fully-supervised  | monocular  |              [code](https://btekin.github.io/)               |
| [Lifting from the Deep: Convolutional 3D Pose Estimation from a Single Image](https://openaccess.thecvf.com/content_cvpr_2017/html/Tome_Lifting_From_the_CVPR_2017_paper.html) | 2017     | weakly-supervised | monocular  | [code](https://github.com/DenisTome/Lifting-from-the-Deep-release) |
| [Towards 3D Human Pose Estimation in the Wild: A Weakly-Supervised Approach](https://openaccess.thecvf.com/content_iccv_2017/html/Zhou_Towards_3D_Human_ICCV_2017_paper.html) | 2017     | weakly-supervised | monocular  |       [code](https://github.com/xingyizhou/pose-hg-3d)       |
| [End-to-End  Recovery of Human Shape and Pose](https://openaccess.thecvf.com/content_cvpr_2018/html/Kanazawa_End-to-End_Recovery_of_CVPR_2018_paper.html) | 2017     | weakly-supervised | monocular  |           [code](https://akanazawa.github.io/hmr/)           |
| [3D Human Pose Estimation in Video With Temporal Convolutions and Semi-Supervised  Training](https://openaccess.thecvf.com/content_CVPR_2019/html/Pavllo_3D_Human_Pose_Estimation_in_Video_With_Temporal_Convolutions_and_CVPR_2019_paper.html) | 2018     | semi-supervised   | monocular  |   [code](https://github.com/facebookresearch/VideoPose3D)   |
| [Ordinal Depth Supervision for 3D Human Pose Estimation](https://openaccess.thecvf.com/content_cvpr_2018/html/Pavlakos_Ordinal_Depth_Supervision_CVPR_2018_paper.html) | 2018     | weakly-supervised | monocular  |    [code](https://github.com/geopavlakos/ordinal-pose3d/)    |
| [Occlusion-Aware Networks for 3D Human Pose Estimation in Video](https://openaccess.thecvf.com/content_ICCV_2019/html/Cheng_Occlusion-Aware_Networks_for_3D_Human_Pose_Estimation_in_Video_ICCV_2019_paper.html) | 2019     | semi-supervised   | monocular  |                              -                               |
| [RepNet: Weakly Supervised Training of an Adversarial Reprojection Network for 3D  Human Pose  Estimation](https://openaccess.thecvf.com/content_CVPR_2019/html/Wandt_RepNet_Weakly_Supervised_Training_of_an_Adversarial_Reprojection_Network_for_CVPR_2019_paper.html) | 2019     | weakly-supervised | monocular  |        [code](https://github.com/bastianwandt/RepNet)        |
| [HoloPose: Holistic 3D Human Reconstruction In-The-Wild](https://openaccess.thecvf.com/content_CVPR_2019/html/Guler_HoloPose_Holistic_3D_Human_Reconstruction_In-The-Wild_CVPR_2019_paper.html) | 2019     | weakly-supervised | monocular  |                              -                               |
| [Multi-task Deep Learning for Real-Time 3D Human Pose Estimation and Action Recognition](https://ieeexplore.ieee.org/abstract/document/9007695) | 2020     | fully-supervised  | monocular  |         [code](https://github.com/dluvizon/deephar)          |
| [3D Human Pose Estimation Using Spatio-Temporal Networks with Explicit Occlusion  Training](https://ojs.aaai.org/index.php/AAAI/article/view/6689) | 2020     | semi-supervised   | monocular  |                              -                               |
| [Multi-View Pose Generator Based on Deep Learning for Monocular 3D Human Pose Estimation](https://www.mdpi.com/2073-8994/12/7/1116) | 2020     | fully-supervised  | monocular  |                              -                               |
| [A Simple Yet Effective Baseline for 3d Human Pose Estimation](https://openaccess.thecvf.com/content_iccv_2017/html/Martinez_A_Simple_yet_ICCV_2017_paper.html) | 2017     | fully-supervised  | multi-view |                              -                               |
| [Self-Supervised Learning of 3D Human Pose Using Multi-View Geometry](https://openaccess.thecvf.com/content_CVPR_2019/html/Kocabas_Self-Supervised_Learning_of_3D_Human_Pose_Using_Multi-View_Geometry_CVPR_2019_paper.html) | 2019     | semi-supervised   | multi-view |      [code](https://github.com/mkocabas/EpipolarPose)       |
| [Learnable Triangulation of Human Pose](https://openaccess.thecvf.com/content_ICCV_2019/html/Iskakov_Learnable_Triangulation_of_Human_Pose_ICCV_2019_paper.html) | 2019     | fully-supervised  | multi-view | [code](https://saic-violet.github.io/learnable-triangulation) |
| [Lightweight Multi-View 3D Pose Estimation through Camera-Disentangled Representation](https://openaccess.thecvf.com/content_CVPR_2020/html/Remelli_Lightweight_Multi-View_3D_Pose_Estimation_Through_Camera-Disentangled_Representation_CVPR_2020_paper.html) | 2020     | fully-supervised  | multi-view |                              -                               |
| [Epipolar Transformers](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Epipolar_Transformers_CVPR_2020_paper.html) | 2020     | weakly-supervised | multi-view |      [code](github.com/yihui-he/epipolar-transformers)       |



### Summary of the state-of-the-arts methods on MPI-INF-3DHP datasets.

| **Title**                                                    | **Year** | **Supervision**         | **Type**   |                       **URL**                        |
| ------------------------------------------------------------ | -------- | ----------------------- | ---------- | :--------------------------------------------------: |
| [Monocular 3D Human Pose Estimation in the Wild Using Improved CNN Supervision](https://ieeexplore.ieee.org/abstract/document/8374605) | 2017     | fully-supervised        | monocular  |   [code](http://gvv.mpi-inf.mpg.de/3dhp-dataset/)    |
| [Towards 3D Human Pose Estimation in the Wild: A Weakly-Supervised Approach](https://openaccess.thecvf.com/content_iccv_2017/html/Zhou_Towards_3D_Human_ICCV_2017_paper.html) | 2017     | weakly-supervised       | monocular  |   [code](https://github.com/xingyizhou/pose-hg-3d)   |
| [3D Human Pose Estimation in the Wild by Adversarial Learning](https://openaccess.thecvf.com/content_cvpr_2018/html/Yang_3D_Human_Pose_CVPR_2018_paper.html) | 2018     | semi-supervised         | monocular  |                          -                           |
| [3d human pose estimation with 2d marginal heatmaps](https://ieeexplore.ieee.org/abstract/document/8658906) | 2019     | weakly-supervised       | monocular  |     [code](https://github.com/anibali/margipose)     |
| [Learning to Reconstruct 3D Human Pose and Shape via Model-Fitting in the Loop](https://openaccess.thecvf.com/content_ICCV_2019/html/Kolotouros_Learning_to_Reconstruct_3D_Human_Pose_and_Shape_via_Model-Fitting_ICCV_2019_paper.html) | 2019     | weakly-supervised       | monocular  | [code](https://seas.upenn.edu/˜nkolot/projects/spin) |
| [RepNet: Weakly Supervised Training of an Adversarial Reprojection Network for 3D  Human Pose  Estimation](https://openaccess.thecvf.com/content_CVPR_2019/html/Wandt_RepNet_Weakly_Supervised_Training_of_an_Adversarial_Reprojection_Network_for_CVPR_2019_paper.html) | 2019     | weakly-supervised       | monocular  |    [code](https://github.com/bastianwandt/RepNet)    |
| [Unsupervised 3D Pose Estimation With Geometric Self-Supervision](https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Unsupervised_3D_Pose_Estimation_With_Geometric_Self-Supervision_CVPR_2019_paper.html) | 2019     | unsupervised-supervised | monocular  |                          -                           |
| [Anatomy-aware 3D Human Pose Estimation with Bone-based Pose Decomposition](https://ieeexplore.ieee.org/abstract/document/9347537/) | 2021     | fully-supervised        | monocular  |  [code](https://github.com/sunnychencool/Anatomy3D)  |
| [Generalizing Monocular 3D Human Pose Estimation in the Wild](https://openaccess.thecvf.com/content_ICCVW_2019/html/GMDL/Wang_Generalizing_Monocular_3D_Human_Pose_Estimation_in_the_Wild_ICCVW_2019_paper.html) | 2019     | weakly-supervised       | multi-view |                          -                           |



### Summary of the state-of-the-art multi-person 3D pose estimation methods on MuPoTS-3D dataset.

| **Title**                                                    | **Year** | **Supervision**    | **Type**  |                           **URL**                            |
| ------------------------------------------------------------ | -------- | ------------------ | --------- | :----------------------------------------------------------: |
| [LCR-Net: Localization-Classification-Regression for Human Pose](https://openaccess.thecvf.com/content_cvpr_2017/html/Rogez_LCR-Net_Localization-Classification-Regression_for_CVPR_2017_paper.html) | 2017     | weakly -supervised | monocular |       [code](https://thoth.inrialpes.fr/src/LCR-Net/)        |
| [Single-shot multi-person 3d pose estimation from monocular rgb](https://ieeexplore.ieee.org/abstract/document/8490962) | 2018     | fully-supervised   | monocular | [code](http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/) |
| [Camera Distance-Aware Top-Down Approach for 3D Multi-Person Pose Estimation From a  Single RGB Image](https://openaccess.thecvf.com/content_ICCV_2019/html/Moon_Camera_Distance-Aware_Top-Down_Approach_for_3D_Multi-Person_Pose_Estimation_From_ICCV_2019_paper.html) | 2019     | fully-supervised   | monocular |  [code](https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE)   |
| [XNect: Real-time Multi-Person 3D Motion Capture with a Single RGB Camera](https://dl.acm.org/doi/abs/10.1145/3386569.3392410) | 2020     | semi-supervised    | monocular |      [code](http://gvv.mpi-inf.mpg.de/projects/XNect/)       |
| [Lcr-net++: Multi-person 2d and 3d pose detection in natural images](https://ieeexplore.ieee.org/abstract/document/8611390) | 2019     | weakly-supervised  | monocular |       [code](https://thoth.inrialpes.fr/src/LCR-Net/)        |
| [Multi-person 3d human pose estimation from monocular images](https://ieeexplore.ieee.org/abstract/document/8886035/) | 2019     | weakly-supervised  | monocular |                              -                               |



### Summary of the state-of-the-art multi-person 3D pose estimation methods on Campus dataset.

| **Title**                                                    | **Year** | **Supervision**   | **Type**   |                           **URL**                            |
| ------------------------------------------------------------ | -------- | ----------------- | ---------- | :----------------------------------------------------------: |
| [3D Pictorial Structures for Multiple Human Pose Estimation](https://openaccess.thecvf.com/content_cvpr_2014/html/Belagiannis_3D_Pictorial_Structures_2014_CVPR_paper.html) | 2014     | fully-supervised  | multi-view |                              -                               |
| [Multiple human pose estimation with temporally consistent 3D pictorial structures](https://link.springer.com/chapter/10.1007/978-3-319-16178-5_52) | 2014     | weakly-supervised | multi-view |                              -                               |
| [3d pictorial structures revisited: Multiple human pose estimation](https://ieeexplore.ieee.org/abstract/document/7360209) | 2015     | fully-supervised  | multi-view |                              -                               |
| [Multiple human 3d pose estimation from multiview images](https://link.springer.com/article/10.1007/s11042-017-5133-8) | 2018     | weakly-supervised | multi-view |                              -                               |
| [Fast and Robust Multi-Person 3D Pose Estimation From Multiple Views](https://openaccess.thecvf.com/content_CVPR_2019/html/Dong_Fast_and_Robust_Multi-Person_3D_Pose_Estimation_From_Multiple_Views_CVPR_2019_paper.html) | 2019     | weakly-supervised | multi-view |          [code](https://zju3dv.github.io/mvpose/)           |
| [Multi-Person 3D Pose Estimation and Tracking in Sports](https://openaccess.thecvf.com/content_CVPRW_2019/html/CVSports/Bridgeman_Multi-Person_3D_Pose_Estimation_and_Tracking_in_Sports_CVPRW_2019_paper.html) | 2019     | unsupervised      | multi-view | [code](https://cvssp.org/projects/4d/multi_person_3d_pose_sports/) |
| [VoxelPose: Towards Multi-camera 3D Human Pose Estimation in Wild Environment](https://link.springer.com/content/pdf/10.1007/978-3-030-58452-8_12.pdf) | 2020     | fully-supervised  | multi-view |    [code](https://github.com/microsoft/voxelpose-pytorch)    |
| [Cross-View Tracking for Multi-Human 3D Pose Estimation at over 100 FPS](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Cross-View_Tracking_for_Multi-Human_3D_Pose_Estimation_at_Over_100_CVPR_2020_paper.html) | 2020     | unsupervised      | multi-view | [code](https://github.com/longcw/crossview_3d_pose_tracking) |



### Summary of the state-of-the-art multi-person 3D pose estimation methods on Shelf dataset.

| **Title**                                                    | **Year** | **Supervision**   | **Type **  |                           **URL**                            |
| ------------------------------------------------------------ | -------- | ----------------- | ---------- | :----------------------------------------------------------: |
| [3D Pictorial Structures for Multiple Human Pose Estimation](https://openaccess.thecvf.com/content_cvpr_2014/html/Belagiannis_3D_Pictorial_Structures_2014_CVPR_paper.html) | 2014     | fully-supervised  | multi-view |                              -                               |
| [Multiple human pose estimation with temporally consistent 3D pictorial structures](https://link.springer.com/chapter/10.1007/978-3-319-16178-5_52) | 2014     | weakly-supervised | multi-view |                              -                               |
| [3d pictorial structures revisited: Multiple human pose estimation](https://ieeexplore.ieee.org/abstract/document/7360209) | 2015     | fully-supervised  | multi-view |                              -                               |
| [Multiple human 3d pose estimation from multiview images](https://link.springer.com/article/10.1007/s11042-017-5133-8) | 2018     | weakly-supervised | multi-view |                              -                               |
| [Fast and Robust Multi-Person 3D Pose Estimation From Multiple Views](https://openaccess.thecvf.com/content_CVPR_2019/html/Dong_Fast_and_Robust_Multi-Person_3D_Pose_Estimation_From_Multiple_Views_CVPR_2019_paper.html) | 2019     | weakly-supervised | multi-view |           [code](https://zju3dv.github.io/mvpose/)           |
| [Multi-Person 3D Pose Estimation and Tracking in Sports](https://openaccess.thecvf.com/content_CVPRW_2019/html/CVSports/Bridgeman_Multi-Person_3D_Pose_Estimation_and_Tracking_in_Sports_CVPRW_2019_paper.html) | 2019     | unsupervised      | multi-view | [code](https://cvssp.org/projects/4d/multi_person_3d_pose_sports/) |
| [VoxelPose: Towards Multi-camera 3D Human Pose Estimation in Wild Environment](https://link.springer.com/content/pdf/10.1007/978-3-030-58452-8_12.pdf) | 2020     | fully-supervised  | multi-view |    [code](https://github.com/microsoft/voxelpose-pytorch)    |
| [Light3DPose: Real-time Multi-Person 3D Pose Estimation from Multiple Views](https://ieeexplore.ieee.org/abstract/document/9412652) | 2021     | weakly-supervised | multi-view |                              -                               |
| [Cross-View Tracking for Multi-Human 3D Pose Estimation at over 100 FPS](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Cross-View_Tracking_for_Multi-Human_3D_Pose_Estimation_at_Over_100_CVPR_2020_paper.html) | 2020     | unsupervised      | multi-view | [code](https://github.com/longcw/crossview_3d_pose_tracking) |



## Acknowledgment

This work is supported by the National National Science Foundation of China (Grant No. 61802355 and 61702350) and the Open Research Project of The Hubei Key Laboratory of Intelligent Geo-Information Processing (KLIGIP-2019B04).
