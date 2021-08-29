# PointwiseNet
## Deep learning methods for 3D human pose estimation under different supervision manners: A survey
3D human pose estimation is an essential technique for many computer vision applications such as video surveillance, human-computer interaction, digital entertainment, etc. Meanwhile, it is also a challenging task due to the inherent ambiguity and occlusion problems. Lately, deep learning method for 3D human pose estimation has attracted increasing attention due to the success of learning based 2D human pose estimation. Against this backdrop, numerous methods have being proposed to address different problems in this area. Generally, these methods consist of the body modeling, learning based pose estimation and regularization for refinement. This paper provides an extensive literature survey of recent literatures about Deep learning methods for 3D human pose estimation, covering the aspects of human body model, 3D single-person pose estimation, 3D multi-person human pose estimation and regularization. Moreover, as deep learning methods can be categorized according to supervision manners which are based on quite different philosophies and are applicable for diverse scenarios, this paper provides a classification for both 3D single-person and multi-person pose estimation methods based on their supervision manners, i.e. unsupervised methods, fully-supervised methods, weakly-supervised methods, semi-supervised methods, and self-supervised methods. At last, this paper also enlists the contemporary and widely used datasets, compares performances of reviewed methods and discusses promising research directions.

**Keywords:** 3D human pose estimation; deep learning; unsupervised; fully-supervised; weakly-supervised; semi-supervised; self-supervised


## Supervision form
With the development of machine learning, researchers divide the supervision into different forms according to variable situations and distinct data types. The supervision falls into five categories:
*unsupervised,* *semi-supervised*, *fully-supervised*, weakly-supervised. In general machine learning, the above five categories have standard definitions. However, definitions could change with the shift of the research scene.

- ***Unsupervised methods*** do not require any multi-view image data, 3D skeletons, correspondences between 2D-3D points, or use previously learned 3D priors during training. Self-supervised methods which can also solve the issue, deficiency of 3D data, have become popular in recent years.
  Self-supervised methods is a form of unsupervised learning where the data provides the supervision.
- ***Fully-supervised methods*** rely on large training sets annotated with ground-truth 3D positions coming from multi-view motion capture systems.
- ***Weakly-supervised methods*** access multiple cues for weak supervision, such as a) paired 2D ground-truth, b) unpaired 3D ground-truth (3D pose without the corresponding image), c) multi-view image pair, d) camera parameters in a multi-view setup, etc.
- ***Semi-supervised methods*** use part of annotated data (e.g. 10 percent of 3D labels), which means labeled training data is scarce.


## The taxonomy of deep learning methods for 3D human pose estimation
Scenes with distinct limitations could yield various approaches with different supervisions. Both 3D single-person pose estimation and 3D multi-person pose estimation combined with different supervision forms could derive various branches as described in Fig.~\ref{fig:taxonomy}.

<img src="https://github.com/djzgroup/PointwiseNet/blob/master/img/part_seg.jpg" width="600">

From Fig.~\ref{fig:taxonomy}, we could observe an unbalanced tree describing deep learning based 3D human pose estimation approaches. 3D multi-person pose estimation has received less interest compared to 3D single-person pose estimation. Also, video-based 3D human pose estimation is less studied than image-based 3D human pose estimation. Another interesting sight is that fully-supervised methods are presented in each sub-category~(i.e.~marked green in Fig.~\ref{fig:taxonomy}), which may indicate that fully-supervised methods are helped to investigate a research area at the beginning. 


## Complexity Analysis
The following table summarizes the space (number of parameters) and the time (floating point operations) complexity of PointwiseNet in 3D object classification task with 1024 points as the input. Compared with PointNet++ [40], PointwiseNet reduces the parameters by 4.7% and the FLOPs by 52.0%, which shows its great potential for real-time applications, e.g., scene parsing in autonomous driving.



| **XXXXXXXXXXXXXXXXXXXXXXXX**                                 | **XXXXXXXXXXXXXXXXXXXXXXXX**                                 | **Year** | **Supervision**   | **Type**   | **URL** |
| ------------------------------------------------------------ | ------------------------------------------------------------ | -------- | ----------------- | ---------- | ------- |
| HM3.6                                                        |                                                              |          |                   |            |         |
| [3d  human pose estimation from monocular images with deep convolutional neural  network]( https://link.springer.com/chapter/10.1007/978-3-319-16808-1_23  ) | https://link.springer.com/chapter/10.1007/978-3-319-16808-1_23 | 2014     | fully-supervised  | monocular  | code    |
| [Sparseness  Meets Deepness: 3D Human Pose Estimation from Monocular Video](https://openaccess.thecvf.com/content_cvpr_2016/html/Zhou_Sparseness_Meets_Deepness_CVPR_2016_paper.html) | https://openaccess.thecvf.com/content_cvpr_2016/html/Zhou_Sparseness_Meets_Deepness_CVPR_2016_paper.html | 2016     | weakly-supervised | monocular  | code    |
| Structured  Prediction of 3D Human Pose with Deep Neural Networks | https://arxiv.org/abs/1605.05180                             | 2016     | fully-supervised  | monocular  | code    |
| [Lifting  from the Deep: Convolutional 3D Pose Estimation from a Single Image]( https://openaccess.thecvf.com/content_cvpr_2017/html/Tome_Lifting_From_the_CVPR_2017_paper.html ) | https://openaccess.thecvf.com/content_cvpr_2017/html/Tome_Lifting_From_the_CVPR_2017_paper.html | 2017     | weakly-supervised | monocular  | code    |
| [Towards  3D Human Pose Estimation in the Wild: A Weakly-Supervised Approach]( https://openaccess.thecvf.com/content_iccv_2017/html/Zhou_Towards_3D_Human_ICCV_2017_paper.html  ) | https://openaccess.thecvf.com/content_iccv_2017/html/Zhou_Towards_3D_Human_ICCV_2017_paper.html | 2017     | weakly-supervised | monocular  | code    |
| [End-to-End  Recovery of Human Shape and Pose](https://openaccess.thecvf.com/content_cvpr_2018/html/Kanazawa_End-to-End_Recovery_of_CVPR_2018_paper.html ) | https://openaccess.thecvf.com/content_cvpr_2018/html/Kanazawa_End-to-End_Recovery_of_CVPR_2018_paper.html | 2017     | weakly-supervised | monocular  | code    |
| [3D  Human Pose Estimation in Video With Temporal Convolutions and Semi-Supervised  Training](https://openaccess.thecvf.com/content_CVPR_2019/html/Pavllo_3D_Human_Pose_Estimation_in_Video_With_Temporal_Convolutions_and_CVPR_2019_paper.html  ) | https://openaccess.thecvf.com/content_CVPR_2019/html/Pavllo_3D_Human_Pose_Estimation_in_Video_With_Temporal_Convolutions_and_CVPR_2019_paper.html | 2018     | semi-supervised   | monocular  | code    |
| [Ordinal  Depth Supervision for 3D Human Pose Estimation](  https://openaccess.thecvf.com/content_cvpr_2018/html/Pavlakos_Ordinal_Depth_Supervision_CVPR_2018_paper.html  ) | https://openaccess.thecvf.com/content_cvpr_2018/html/Pavlakos_Ordinal_Depth_Supervision_CVPR_2018_paper.html | 2018     | weakly-supervised | monocular  | code    |
| [Occlusion-Aware  Networks for 3D Human Pose Estimation in Video](  https://openaccess.thecvf.com/content_ICCV_2019/html/Cheng_Occlusion-Aware_Networks_for_3D_Human_Pose_Estimation_in_Video_ICCV_2019_paper.html  ) | https://openaccess.thecvf.com/content_ICCV_2019/html/Cheng_Occlusion-Aware_Networks_for_3D_Human_Pose_Estimation_in_Video_ICCV_2019_paper.html | 2019     | semi-supervised   | monocular  | code    |
| [RepNet:  Weakly Supervised Training of an Adversarial Reprojection Network for 3D  Human Pose  Estimation](https://openaccess.thecvf.com/content_CVPR_2019/html/Wandt_RepNet_Weakly_Supervised_Training_of_an_Adversarial_Reprojection_Network_for_CVPR_2019_paper.html) | https://openaccess.thecvf.com/content_CVPR_2019/html/Wandt_RepNet_Weakly_Supervised_Training_of_an_Adversarial_Reprojection_Network_for_CVPR_2019_paper.html | 2019     | weakly-supervised | monocular  | code    |
| [HoloPose:  Holistic 3D Human Reconstruction In-The-Wild](https://openaccess.thecvf.com/content_CVPR_2019/html/Guler_HoloPose_Holistic_3D_Human_Reconstruction_In-The-Wild_CVPR_2019_paper.html  ) | https://openaccess.thecvf.com/content_CVPR_2019/html/Guler_HoloPose_Holistic_3D_Human_Reconstruction_In-The-Wild_CVPR_2019_paper.html | 2019     | weakly-supervised | monocular  | code    |
| [Multi-task  Deep Learning for Real-Time 3D Human Pose Estimation and Action Recognition](https://ieeexplore.ieee.org/abstract/document/9007695 ) | https://ieeexplore.ieee.org/abstract/document/9007695        | 2020     | fully-supervised  | monocular  | code    |
| [3D  Human Pose Estimation Using Spatio-Temporal Networks with Explicit Occlusion  Training](https://ojs.aaai.org/index.php/AAAI/article/view/6689  ) | https://ojs.aaai.org/index.php/AAAI/article/view/6689        | 2020     | semi-supervised   | monocular  | code    |
| [A  Simple Yet Effective Baseline for 3d Human Pose Estimation](https://openaccess.thecvf.com/content_iccv_2017/html/Martinez_A_Simple_yet_ICCV_2017_paper.html ) | https://openaccess.thecvf.com/content_iccv_2017/html/Martinez_A_Simple_yet_ICCV_2017_paper.html | 2017     | fully-supervised  | multi-view | code    |
| [Self-Supervised  Learning of 3D Human Pose Using Multi-View Geometry](https://openaccess.thecvf.com/content_CVPR_2019/html/Kocabas_Self-Supervised_Learning_of_3D_Human_Pose_Using_Multi-View_Geometry_CVPR_2019_paper.html) | https://openaccess.thecvf.com/content_CVPR_2019/html/Kocabas_Self-Supervised_Learning_of_3D_Human_Pose_Using_Multi-View_Geometry_CVPR_2019_paper.html | 2019     | semi-supervised   | multi-view | code    |
| [Learnable  Triangulation of Human Pose]( https://openaccess.thecvf.com/content_ICCV_2019/html/Iskakov_Learnable_Triangulation_of_Human_Pose_ICCV_2019_paper.html ) | https://openaccess.thecvf.com/content_ICCV_2019/html/Iskakov_Learnable_Triangulation_of_Human_Pose_ICCV_2019_paper.html | 2019     | fully-supervised  | multi-view | code    |
| [Lightweight  Multi-View 3D Pose Estimation through Camera-Disentangled Representation](https://openaccess.thecvf.com/content_CVPR_2020/html/Remelli_Lightweight_Multi-View_3D_Pose_Estimation_Through_Camera-Disentangled_Representation_CVPR_2020_paper.html ) | https://openaccess.thecvf.com/content_CVPR_2020/html/Remelli_Lightweight_Multi-View_3D_Pose_Estimation_Through_Camera-Disentangled_Representation_CVPR_2020_paper.html | 2020     | fully-supervised  | multi-view | code    |
| [Epipolar  Transformers](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Epipolar_Transformers_CVPR_2020_paper.html  ) | https://openaccess.thecvf.com/content_CVPR_2020/html/He_Epipolar_Transformers_CVPR_2020_paper.html | 2020     | weakly-supervised | multi-view | code    |



| **XXXXXXXXXXXXXXXXXXXXXXXX**                                 | **XXXXXXXXXXXXXXXXXXXXXXXX**                                 | **Year** | **Supervision**         | **Type**   | **URL** |
| ------------------------------------------------------------ | ------------------------------------------------------------ | -------- | ----------------------- | ---------- | ------- |
| MPI-INF-3DHP                                                 |                                                              |          |                         |            |         |
| Monocular  3D Human Pose Estimation in the Wild Using Improved CNN Supervision | https://ieeexplore.ieee.org/abstract/document/8374605        | 2017     | fully-supervised        | monocular  | code    |
| Towards  3D Human Pose Estimation in the Wild: A Weakly-Supervised Approach | https://openaccess.thecvf.com/content_iccv_2017/html/Zhou_Towards_3D_Human_ICCV_2017_paper.html | 2017     | weakly-supervised       | monocular  | code    |
| 3D  Human Pose Estimation in the Wild by Adversarial Learning | https://openaccess.thecvf.com/content_cvpr_2018/html/Yang_3D_Human_Pose_CVPR_2018_paper.html | 2018     | semi-supervised         | monocular  | code    |
| 3d  human pose estimation with 2d marginal heatmaps          | https://ieeexplore.ieee.org/abstract/document/8658906        | 2019     | weakly-supervised       | monocular  | code    |
| Learning to Reconstruct 3D  Human Pose and Shape via Model-Fitting in the Loop | https://openaccess.thecvf.com/content_ICCV_2019/html/Kolotouros_Learning_to_Reconstruct_3D_Human_Pose_and_Shape_via_Model-Fitting_ICCV_2019_paper.html | 2019     | weakly-supervised       | monocular  | code    |
| RepNet:  Weakly Supervised Training of an Adversarial Reprojection Network for 3D  Human Pose  Estimation | https://openaccess.thecvf.com/content_CVPR_2019/html/Wandt_RepNet_Weakly_Supervised_Training_of_an_Adversarial_Reprojection_Network_for_CVPR_2019_paper.html | 2019     | weakly-supervised       | monocular  | code    |
| Unsupervised  3D Pose Estimation With Geometric Self-Supervision | https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Unsupervised_3D_Pose_Estimation_With_Geometric_Self-Supervision_CVPR_2019_paper.html | 2019     | unsupervised-supervised | monocular  | code    |
| Anatomy-aware  3D Human Pose Estimation with Bone-based Pose Decomposition | https://arxiv.org/abs/2002.10322                             | 2021     | fully-supervised        | monocular  | code    |
| Generalizing  Monocular 3D Human Pose Estimation in the Wild | https://openaccess.thecvf.com/content_ICCVW_2019/html/GMDL/Wang_Generalizing_Monocular_3D_Human_Pose_Estimation_in_the_Wild_ICCVW_2019_paper.html | 2019     | weakly-supervised       | multi-view | code    |



| **XXXXXXXXXXXXXXXXXXXXXXXX**                                 | **XXXXXXXXXXXXXXXXXXXXXXXX**                                 | **Year** | **Supervision**    | **Type**  | **URL** |
| ------------------------------------------------------------ | ------------------------------------------------------------ | -------- | ------------------ | --------- | ------- |
| MuPoTs-3D                                                    |                                                              |          |                    |           |         |
| LCR-Net:  Localization-Classification-Regression for Human Pose | https://openaccess.thecvf.com/content_cvpr_2017/html/Rogez_LCR-Net_Localization-Classification-Regression_for_CVPR_2017_paper.html | 2017     | weakly -supervised | monocular | code    |
| Single-shot  multi-person 3d pose estimation from monocular rgb | https://ieeexplore.ieee.org/abstract/document/8490962        | 2018     | fully-supervised   | monocular | code    |
| Camera  Distance-Aware Top-Down Approach for 3D Multi-Person Pose Estimation From a  Single RGB Image | https://openaccess.thecvf.com/content_ICCV_2019/html/Moon_Camera_Distance-Aware_Top-Down_Approach_for_3D_Multi-Person_Pose_Estimation_From_ICCV_2019_paper.html | 2019     | fully-supervised   | monocular | code    |
| XNect:  Real-time Multi-Person 3D Motion Capture with a Single RGB Camera | https://dl.acm.org/doi/abs/10.1145/3386569.3392410           | 2020     | semi-supervised    | monocular | code    |
| Lcr-net++:  Multi-person 2d and 3d pose detection in natural images | https://ieeexplore.ieee.org/abstract/document/8611390        | 2019     | weakly-supervised  | monocular | code    |
| Multi-person  3d human pose estimation from monocular images | https://ieeexplore.ieee.org/abstract/document/8886035/       | 2019     | weakly-supervised  | monocular | code    |





| **XXXXXXXXXXXXXXXXXXXXXXXX**                                 | **XXXXXXXXXXXXXXXXXXXXXXXX**                                 | **Year** | **Supervision**   | **Type**   | **URL** |
| ------------------------------------------------------------ | ------------------------------------------------------------ | -------- | ----------------- | ---------- | ------- |
| Campus                                                       |                                                              |          |                   |            |         |
| 3D  Pictorial Structures for Multiple Human Pose Estimation  | https://openaccess.thecvf.com/content_cvpr_2014/html/Belagiannis_3D_Pictorial_Structures_2014_CVPR_paper.html | 2014     | fully-supervised  | multi-view | code    |
| Multiple  human pose estimation with temporally consistent 3D pictorial structures | https://link.springer.com/chapter/10.1007/978-3-319-16178-5_52 | 2014     | weakly-supervised | multi-view | code    |
| 3d  pictorial structures revisited: Multiple human pose estimation | https://ieeexplore.ieee.org/abstract/document/7360209        | 2015     | fully-supervised  | multi-view | code    |
| Multiple  human 3d pose estimation from multiview images     | https://link.springer.com/article/10.1007/s11042-017-5133-8  | 2018     | weakly-supervised | multi-view | code    |
| Fast  and Robust Multi-Person 3D Pose Estimation From Multiple Views | https://openaccess.thecvf.com/content_CVPR_2019/html/Dong_Fast_and_Robust_Multi-Person_3D_Pose_Estimation_From_Multiple_Views_CVPR_2019_paper.html | 2019     | weakly-supervised | multi-view | code    |
| Multi-Person  3D Pose Estimation and Tracking in Sports      | https://openaccess.thecvf.com/content_CVPRW_2019/html/CVSports/Bridgeman_Multi-Person_3D_Pose_Estimation_and_Tracking_in_Sports_CVPRW_2019_paper.html | 2019     | unsupervised      | multi-view | code    |
|                                                              |                                                              | 2020     | fully-supervised  | multi-view | code    |
| Cross-View  Tracking for Multi-Human 3D Pose Estimation at over 100 FPS | https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Cross-View_Tracking_for_Multi-Human_3D_Pose_Estimation_at_Over_100_CVPR_2020_paper.html | 2020     | unsupervised      | multi-view | code    |







| **XXXXXXXXXXXXXXXXXXXXXXXX**                                 | **XXXXXXXXXXXXXXXXXXXXXXXX**                                 | **Year** | **Supervision**   | **Type**   | **URL** |
| ------------------------------------------------------------ | ------------------------------------------------------------ | -------- | ----------------- | ---------- | ------- |
| Shelf                                                        |                                                              |          |                   |            |         |
| 3D  Pictorial Structures for Multiple Human Pose Estimation  | https://openaccess.thecvf.com/content_cvpr_2014/html/Belagiannis_3D_Pictorial_Structures_2014_CVPR_paper.html | 2014     | fully-supervised  | multi-view | code    |
| Multiple  human pose estimation with temporally consistent 3D pictorial structures | https://link.springer.com/chapter/10.1007/978-3-319-16178-5_52 | 2014     | weakly-supervised | multi-view | code    |
| 3d  pictorial structures revisited: Multiple human pose estimation | https://ieeexplore.ieee.org/abstract/document/7360209        | 2015     | fully-supervised  | multi-view | code    |
| Multiple  human 3d pose estimation from multiview images     | https://link.springer.com/article/10.1007/s11042-017-5133-8  | 2018     | weakly-supervised | multi-view | code    |
| Fast  and Robust Multi-Person 3D Pose Estimation From Multiple Views | https://openaccess.thecvf.com/content_CVPR_2019/html/Dong_Fast_and_Robust_Multi-Person_3D_Pose_Estimation_From_Multiple_Views_CVPR_2019_paper.html | 2019     | weakly-supervised | multi-view | code    |
| Multi-Person  3D Pose Estimation and Tracking in Sports      | https://openaccess.thecvf.com/content_CVPRW_2019/html/CVSports/Bridgeman_Multi-Person_3D_Pose_Estimation_and_Tracking_in_Sports_CVPRW_2019_paper.html | 2019     | unsupervised      | multi-view | code    |
|                                                              |                                                              | 2020     | fully-supervised  | multi-view | code    |
| Light3DPose:  Real-time Multi-Person 3D Pose Estimation from Multiple Views | https://openaccess.thecvf.com/content_CVPRW_2019/html/CVSports/Bridgeman_Multi-Person_3D_Pose_Estimation_and_Tracking_in_Sports_CVPRW_2019_paper.html | 2021     | weakly-supervised | multi-view | code    |
| Cross-View  Tracking for Multi-Human 3D Pose Estimation at over 100 FPS | https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Cross-View_Tracking_for_Multi-Human_3D_Pose_Estimation_at_Over_100_CVPR_2020_paper.html | 2020     | unsupervised      | multi-view | code    |



## References

- [1] Qi C R, Su H, Mo K, et al. Pointnet: Deep learning on point sets for 3d classification and segmentation[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017: 652-660.
- [2] Qi C R, Yi L, Su H, et al. Pointnet++: Deep hierarchical feature learning on point sets in a metric space[C]//Advances in neural information processing systems. 2017: 5099-5108.
- [3] Ben-Shabat Y, Lindenbaum M, Fischer A. 3dmfv: Three-dimensional point cloud classification in real-time using convolutional neural networks[J]. IEEE Robotics and Automation Letters, 2018, 3(4): 3145-3152.
- [4] Li J, Chen B M, Hee Lee G. So-net: Self-organizing network for point cloud analysis[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 9397-9406.

## Acknowledgment
This work was supported in part by the National Natural Science Foundation of China under Grant 61702350 and Grant 61472289 and in part by the Open Project Program of the State Key Laboratory of Digital Manufacturing Equipment and Technology, HUST, under Grant DMETKF2017016.
