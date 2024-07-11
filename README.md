# Adaptive KalmanNet

In real-world applications of filtering, different systems may have different State Space Model (SSM) parameters. However, neural network aided filters are usually trained on fixed or limited number of SSMs, where generalizing to different SSMs needs time-consuming and computationally intensive retraining. 

This work targets at the shifts in evolution and observation noise distributions. It is based on [KalmanNet](https://arxiv.org/abs/2107.10043), and enables KalmanNet with the fast adaptation ability to shifting noise distributions during inference.

Paper Link: https://arxiv.org/abs/2309.07016 (published on ICASSP 2024)
