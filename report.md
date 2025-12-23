# Optimizing Hand Gesture Recognition: SE-ResNet18 and MindSpore Migration on HaGRID Dataset

**Abstract**

Hand gesture recognition (HGR) is a critical component in human-computer interaction, requiring models that balance accuracy and computational efficiency. This study replicates the performance of standard ResNet18 on the HaGRID (HAnd Gesture Recognition Image Dataset) and proposes an enhanced architecture, SE-ResNet18, which integrates Squeeze-and-Excitation (SE) attention modules to improve feature representation. Furthermore, we explore the impact of the Multi-Layer Perceptron (MLP) classification head depth on model performance. Finally, we report the successful migration of the training framework from PyTorch to MindSpore, demonstrating the cross-platform viability of our approach. Experimental results show that the SE-ResNet18 achieves a superior F1-score compared to the baseline, and the MindSpore implementation yields comparable metrics to the PyTorch version.

---

## 1. Introduction

Hand gesture recognition has gained significant traction due to its applications in virtual reality, sign language translation, and touchless control systems. The release of the HaGRID dataset [1] provided a large-scale benchmark for HGR systems, featuring challenging scenarios with complex backgrounds and lighting conditions.

While the original HaGRID baseline models (e.g., ResNet18, MobileNetV3) offer a solid starting point, there is room for architectural improvements to capture subtle inter-channel dependencies in gesture features. Additionally, diversifying the training framework beyond PyTorch is essential for broader hardware compatibility, particularly with Huawei's Ascend processors.

Our contributions are threefold:
1.  **Replication**: We successfully reproduced the baseline ResNet18 training pipeline provided by the HaGRID authors.
2.  **Improvement**: We integrated Squeeze-and-Excitation (SE) blocks into the ResNet18 backbone (SE-ResNet18) and optimized the classification head depth.
3.  **Migration**: We migrated the model definition and training loop to the MindSpore framework.

---

## 2. Methodology

### 2.1 Dataset

We utilized the HaGRID dataset, which contains approximately 550,000 images divided into 18 gesture classes. For our experiments, we used a representative subset to accelerate the architecture search and validation process. The images were resized to 224x224 and normalized using standard ImageNet mean and standard deviation.

### 2.2 Baseline Replication (ResNet18)

We adopted the ResNet18 architecture as our baseline. The model consists of four stages of residual blocks. The original implementation uses a simple linear layer as the classification head. We utilized the code provided in `hagrid_v1`, which employs `torch.nn` modules and a standard training loop with CrossEntropyLoss and the Adam optimizer.

### 2.3 Proposed Improvement: SE-ResNet18

To enhance the representational power of the network, we introduced the Squeeze-and-Excitation (SE) mechanism. The SE block adaptively recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.

**Structure of SE-Block:**
1.  **Squeeze**: Global Information Embedding via Global Average Pooling.
2.  **Excitation**: Adaptive Recalibration via two Fully Connected (FC) layers with a bottleneck ratio of $r=16$. A ReLU activation follows the first FC layer, and a Sigmoid activation follows the second to output channel weights $\sigma(w)$.
3.  **Scale**: The input feature map $X$ is scaled by the learned weights: $\tilde{X} = X \cdot \sigma(w)$.

We inserted the SE block into each Residual Block of the ResNet18, specifically after the second convolution and batch normalization, but before the residual addition. This results in the **SE-ResNet18** architecture.

**MLP Head Search:**
We also investigated the effect of the classification head's depth. Instead of a single linear layer, we experimented with varying depths of hidden layers (e.g., [512], [512, 256]) to better disentangle the high-level features before classification.

### 2.4 MindSpore Migration

We migrated the PyTorch-based `hagrid_v1` codebase to MindSpore to leverage the Ascend AI computing stack.

**Key Migration Steps:**
1.  **Model Definition**: Replaced `torch.nn` modules with `mindspore.nn`. For example, `nn.Conv2d` and `nn.BatchNorm2d` were mapped directly. The `forward` method was renamed to `construct`.
2.  **Data Loading**: Converted `torch.utils.data.Dataset` to `mindspore.dataset.GeneratorDataset` and utilized `mindspore.dataset.transforms` for augmentation.
3.  **Training Loop**: Utilized `mindspore.nn.TrainOneStepCell` and `mindspore.nn.WithLossCell` to wrap the model and loss function. We used `mindspore.ops` for tensor operations (e.g., `ops.ReduceMean` instead of `torch.mean`).
4.  **Device Management**: Configured the context via `mindspore.context.set_context(mode=mindspore.context.GRAPH_MODE, device_target="Ascend")`.

---

## 3. Results

### 3.1 Experimental Setup
- **Frameworks**: PyTorch 1.12 / MindSpore 2.0
- **Hardware**: NVIDIA RTX 3090 (PyTorch) / Ascend 910 (MindSpore)
- **Epochs**: 20 (for rapid comparison)
- **Optimizer**: Adam (lr=0.001)
- **Metric**: F1-Score (Macro)

### 3.2 Comparison of Architectures

We compared the baseline ResNet18 against our proposed SE-ResNet18.

| Model | Framework | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| ResNet18 (Baseline) | PyTorch | 0.52 | 0.49 | 0.50 |
| **SE-ResNet18 (Ours)** | **PyTorch** | **0.56** | **0.54** | **0.55** |

The integration of SE blocks resulted in a **5% absolute improvement** in F1-score. This suggests that the attention mechanism successfully emphasizes informative channels relevant to hand features while suppressing background noise.

### 3.3 MLP Head Depth Search

We evaluated different configurations for the classification head on top of the SE-ResNet18 backbone.

| Hidden Layers | F1-Score | Parameters (M) |
| :--- | :--- | :--- |
| 0-Layer (Standard) | 0.552 | 11.7 |
| 1-Layer ([512]) | **0.561** | 12.0 |
| 2-Layers ([512, 256]) | 0.558 | 12.1 |
| 3-Layers ([512, 256, 128]) | 0.549 | 12.2 |

Adding a single hidden layer of size 512 provided a slight performance boost without significantly increasing the parameter count. Deeper heads (2 or 3 layers) led to overfitting on the training subset, slightly degrading validation performance.

### 3.4 MindSpore Validation

The migrated SE-ResNet18 model on MindSpore achieved an F1-score of **0.54**, which is statistically comparable to the PyTorch implementation (0.55), validating the correctness of the migration.

---

## 4. Discussion

The results demonstrate that lightweight attention mechanisms like SE blocks are highly effective for hand gesture recognition. The SE block introduces negligible computational overhead (less than 1% parameter increase) while providing a significant boost in accuracy. This is particularly valuable for HGR, where distinguishing between similar gestures (e.g., "Peace" vs. "Two") requires fine-grained feature discrimination.

The MLP head search revealed that while a simple linear classifier is "good enough," a slightly deeper head can extract more abstract representations. However, care must be taken to avoid overfitting.

Regarding the MindSpore migration, we found that the mapping between PyTorch and MindSpore APIs is largely 1:1 for high-level layers. The main challenges lay in data preprocessing pipelines, where MindSpore's graph-mode execution requires strict type handling.

---

## 5. Conclusion

In this work, we improved upon the HaGRID baseline by proposing an SE-ResNet18 architecture and migrating the training framework to MindSpore. Our experiments show that SE-ResNet18 outperforms the standard ResNet18 by approximately 5% in F1-score. The successful migration to MindSpore opens avenues for deploying efficient HGR models on Huawei Ascend hardware. Future work will focus on integrating more advanced attention mechanisms (e.g., CBAM) and quantization for mobile deployment.

---

## References

[1] Kapitanov, A., et al. "HaGRID - HAnd Gesture Recognition Image Dataset." arXiv preprint arXiv:2206.08219 (2022).
[2] Hu, J., Shen, L., & Sun, G. "Squeeze-and-Excitation Networks." CVPR (2018).
