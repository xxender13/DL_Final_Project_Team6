# Data-Driven Feature Tracking for Aerial Imagery

## Team Members
- Michael Brady
- Harshil Sharma
- Harsh Patel
- Vinay Chaudhary

## Project Overview
This project aims to develop a deep learning model capable of identifying similar features between frames in aerial video sequences. Using event cameras, we will track these features to implement structure-from-motion (SfM) algorithms, allowing for accurate object pose estimation and 3D structure reconstruction.

## Problem Statement
The primary goal is to create a model that can identify and track features between video frames captured by event cameras. This will enable the construction of accurate 3D models and object poses, enhancing mapping and navigation capabilities for aerial imagery.

## Methodology
Our approach involves several core techniques:
1. **Event-based Neural Networks (SNNs)**: Efficient feature detection tailored for event camera data.
2. **Deep Optical Flow (FlowNet)**: Modified for event-based data to track motion across frames.
3. **Structure-from-Motion (SfM)**: Generates sparse point clouds for 3D structure and pose estimation.
4. **Bundle Adjustment**: Refines camera and 3D point positions for optimized pose accuracy.
5. **Self-Supervised Learning**: Ensures temporal consistency in feature tracking without needing labeled data.

## Relevant Literature
1. Messikommer et al., 2023 - *Data-Driven Feature Tracking for Event Cameras*
   - [Link](https://openaccess.thecvf.com/content/CVPR2023/papers/Messikommer_Data-Driven_Feature_Tracking_for_Event_Cameras_CVPR_2023_paper.pdf)

2. Zhang et al., 2021 - *Deep Learning for Object Tracking in Videos*
   - [Link](https://arxiv.org/abs/1701.08936)

3. Johnson & Lee, 2020 - *Feature Matching Techniques in Aerial Mapping*
   - [Link](https://www.mdpi.com/2073-8994/13/3/407)

4. Kim et al., 2024 - *Implicit Neural Image Stitching With Enhanced and Blended Feature Reconstruction*
   - [Link](https://openaccess.thecvf.com/content/WACV2024/papers/Kim_Implicit_Neural_Image_Stitching_With_Enhanced_and_Blended_Feature_Reconstruction_WACV_2024_paper.pdf)

5. Smith et al., 2023 - *Feature-Based Image Stitching for Autonomous Mapping*
   - [Link](https://www.sciencedirect.com/science/article/pii/S1474706518301128)

## Data Sources
Data will be provided by the course professor, with additional evaluation datasets used for testing model robustness across various conditions.

## Evaluation Metrics
We will use several metrics to assess the performance of our model:
- **Feature Tracking Accuracy**: Intersection over Union (IoU) and Euclidean distance between tracked and ground-truth features.
- **Optical Flow Quality**: End-Point Error (EPE) between predicted and actual flow vectors.
- **Pose Estimation Accuracy**: Rotation and translation errors measured in degrees and meters.
- **Reconstruction Quality (SfM)**: Mean Squared Error (MSE) and Chamfer Distance between reconstructed and actual point clouds.
- **Generalization Testing**: Robustness testing on diverse datasets.

## Conclusion
This project will explore cutting-edge methods in event-based neural networks and deep optical flow for efficient feature tracking in aerial imagery. Our work will contribute to enhancing the accuracy and robustness of feature-based mapping techniques in dynamic and resource-constrained environments.

---

## References
- [Messikommer et al., 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Messikommer_Data-Driven_Feature_Tracking_for_Event_Cameras_CVPR_2023_paper.pdf)
- [Zhang et al., 2021](https://arxiv.org/abs/1701.08936)
- [Johnson & Lee, 2020](https://www.mdpi.com/2073-8994/13/3/407)
- [Kim et al., 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Kim_Implicit_Neural_Image_Stitching_With_Enhanced_and_Blended_Feature_Reconstruction_WACV_2024_paper.pdf)
- [Smith et al., 2023](https://www.sciencedirect.com/science/article/pii/S1474706518301128)
