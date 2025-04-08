# Report 1

Describe what you have learned during this study period. You can also include references to external material you have used to enhance your learning process (e.g., book chapters, scientific papers, online courses, online sites).

| Task     | Time Spent (hours) |
| -------- | ------------------ |
| Lectures | 2                  |
| Labs     | 4                  |

## Deeper into CNN

Convolutional neural networks (CNN) are specialized neural networks designed to handle data with grid-like structures, such as images and time-series data. They use convolution operations, which combine input data with a small matrix called a kernel to produce feature maps, offering significant efficiency benefits over traditional methods.

| Term                        | Definition                                                           |
| --------------------------- | -------------------------------------------------------------------- |
| Convolution                 | Linear operation combining input with kernel to produce feature map. |
| Kernel/Filter               | Small matrix slid over input for convolution.                        |
| Feature Map                 | Output of convolution, representing detected features.               |
| Pooling                     | Operation reducing spatial dimensions, e.g., max or average pooling. |
| Sparse Interactions         | Output connects to small input region, reducing computations.        |
| Parameter Sharing           | Same kernel used across input, reducing parameters.                  |
| Equivariant Representations | Outputs transform predictably with input shifts.                     |
| Stride                      | Step size for kernel movement, affecting output size.                |
| Padding                     | Adds zeros around input, handling edge effects in convolution.       |

## Tesla Vision

I’m deeply interested in autonomous vehicles, especially in how Tesla approaches the development of self-driving systems through large-scale data collection, neural network training, and real-world deployment. Tesla’s vision-based strategy—relying on cameras rather than lidar—fascinates me, as it pushes the boundaries of what deep learning can achieve in dynamic, unpredictable environments.

https://www.youtube.com/live/j0z4FweCy4M?feature=shared&t=2928

![alt text](image-1.png)
![alt text](image.png)
![alt text](image-2.png)
![alt text](image-3.png)

### Feature Pyramid Network

https://arxiv.org/pdf/1612.03144v2

Feature Pyramid Networks are a cornerstone in modern computer vision, particularly for tasks like object detection where objects appear at varying scales in an image. FPNs address a fundamental challenge: how to efficiently detect objects of different sizes without sacrificing computational efficiency or accuracy. FPNs build on the natural hierarchical structure of convolutional neural networks (CNNs). In a typical CNN, as you move deeper through the layers, the spatial resolution of feature maps decreases (due to pooling or striding), while the semantic richness—information about what objects are—increases. However, shallow layers retain high spatial resolution but lack strong semantic understanding, making it hard to detect small objects using only deep layers or large objects using only shallow layers. FPNs solve this by creating a pyramid of feature maps that combine the best of both worlds.

#### Bottom-Up Pathway

This is the standard forward pass of a CNN (like ResNet), where feature maps are computed at multiple scales. For example, early layers might output high-resolution maps (e.g., 160x120), while deeper layers produce low-resolution maps (e.g., 20x15) with richer semantics.

#### Top-Down Pathway

FPN takes the deepest, most semantically rich feature map and upsamples it (increases its resolution) to match the scale of shallower layers. This process "hallucinates" higher-resolution features with strong semantics.

#### Lateral Connections

At each level, the upsampled features from the top-down pathway are merged with corresponding feature maps from the bottom-up pathway (which have the same spatial size). This fusion is typically done via element-wise addition after a 1x1 convolution to align channel dimensions. The result is a set of feature maps at multiple scales, each with both high resolution and strong semantic content.

### RegNet

https://arxiv.org/pdf/2003.13678

Regular Networks (RegNets) are a family of efficient CNN architectures designed by Facebook AI Research. They aim to optimize the trade-off between accuracy and computational efficiency, making them ideal for resource-constrained environments. The key idea is to create a "regular" network structure where parameters like width (number of channels), depth (number of layers), and resolution are systematically varied and optimized

#### Parameterized Design

RegNets use a template where the network is divided into stages. Each stage has a fixed number of blocks, and parameters like channel width increase linearly or exponentially across stages.

#### Efficiency

RegNets are designed to maximize accuracy per unit of computation (e.g., FLOPs)

## Oral Exam

1. What is a convolution operation?

    A convolution operation is a mathematical process used in CNNs to extract features from input data, such as images or time-series, by combining the input with a smaller matrix called a kernel (or filter). It involves sliding the kernel over the input, performing element-wise multiplication between the kernel and the overlapping input region, and summing the results to produce a single output value. This process is repeated across the entire input to generate a feature map, which highlights features like edges, textures, or patterns.

    | Definition | Formula                                                                                                     |
    | ---------- | ----------------------------------------------------------------------------------------------------------- |
    | Continuous | $$ s(t) = \int x(a)w(t-a)da $$ Denoted as $s(t) = (x * w)(t)$, where $x$ is the input and $w$ is the kernel |
    | 1D         | $$ s(t) = \sum\_{a=-\infty}^{\infty} x(a)w(t-a) $$                                                          |
    | 2D         | $$ S(i,j) = \sum\_{m,n} I(m,n)K(i-m,j-n) $$ Where $I$ is the input image and $K$ is the kernel              |

2. What is a full, same and valid kernel?

    Valid Convolution involes no padding. The kernel slides only where it fully overlaps the input.

    Same Convolution involes padding to ensure output size matches input size (stride=1).

    Full Convolution uses padding to allow the kernel to slide beyond input edges, increasing output size.

    |          | Valid                                                                                                                     | Same                                                                                                                                               | Full                                                                                                                      |
    | -------- | ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
    | Output   | $$ W*{\text{out}} = W*{\text{in}} - W*{\text{kernel}} + 1 $$ $$ H*{\text{out}} = H*{\text{in}} - H*{\text{kernel}} + 1 $$ | $$ W*{\text{out}} = W*{\text{in}}, \quad H*{\text{out}} = H*{\text{in}} $$                                                                         | $$ W*{\text{out}} = W*{\text{in}} + W*{\text{kernel}} - 1 $$ $$ H*{\text{out}} = H*{\text{in}} + H*{\text{kernel}} - 1 $$ |
    | Example  | Input $32\times32$, kernel $3\times3$ → Output $30\times30$                                                               | $$ P = \frac{W\_{\text{kernel}} - 1}{2}  \space (\text{for odd-sized kernels})$$  Input $32\times32$, kernel $3\times3$, padding 1 → Output $32\times32$ | Input $32\times32$, kernel $3\times3$ → Output $34\times34$                                                               |
    | Use Case | Reduces dimensionality                                                                                                    | Preserves spatial dimensions                                                                                                                       | Rare in CNNs, used in signal processing                                                                                   |

3. What is the difference between convolution and correlation?

    Convolution and correlation are both mathematical operations used to measure the similarity or interaction between two signals, but they differ in how one of the signals is processed. In correlation, one signal is slid over another without flipping, and at each step, the overlapping values are multiplied and summed to measure similarity—this is often used in pattern recognition and signal matching. In convolution, however, one of the signals (usually called the kernel or filter) is flipped before being slid across the other; this flipping makes convolution associative and more suitable for systems analysis, such as in linear time-invariant (LTI) systems.

    PyTorch uses correlation imeplmentation.

    | Correlation                                 | Convolution                                   |
    | ------------------------------------------- | --------------------------------------------- |
    | $$ S(i,j) = \sum\_{m,n} I(i+m,j+n)K(m,n) $$ | $$ S(i,j) = \sum\_{m,n} I(m,n)K(i-m,j-n) $$   |
    | No flipping; kernel applied as-is           | Flips the kernel (180° in 2-D) before sliding |
    | Not commutative ($I * K \neq K * I$)        | Commutative ($I * K = K * I$)                 |
    | Direct similarity between input and kernel  | Similarity with a reversed kernel             |

4. How does backpropagation work in CNNs?

    Backpropagation in Convolutional Neural Networks (CNNs) is the process of updating the network’s weights by calculating the gradient of the loss function with respect to each weight through the chain rule. In CNNs, this involves not just fully connected layers, but also convolutional layers and pooling layers. During the forward pass, each layer computes feature maps by applying filters and non-linear activations. In the backward pass, the gradient of the loss is propagated from the output layer back to the earlier layers. For convolutional layers, the gradients are computed with respect to both the filter weights and the input feature maps, considering how each output depends on overlapping regions of the input. The weights of the filters are then updated using optimization techniques like stochastic gradient descent. Pooling layers typically pass the gradients only to the input neurons that contributed to the pooled output. 

5. What is transfer learning and when is it applied?

    Transfer Learning reuses a model trained on one task for a related task, common in CNNs with pre-trained networks (e.g., on ImageNet).
