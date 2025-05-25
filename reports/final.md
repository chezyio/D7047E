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

# Report 2

Describe what you have learned during this study period. You can also include references to external material you have used to enhance your learning process (e.g., book chapters, scientific papers, online courses, online sites).

| Task     | Time Spent (hours) |
| -------- | ------------------ |
| Lectures | 4                  |
| Labs     | 6                  |

## Generative Modelling

This week, I delved into foundational concepts related to probability distribution learning and density estimation. The primary goal was to understand how to train models that can represent the underlying probability distribution of a given dataset. These models allow the generation of new data instances that resemble the original data by effectively capturing the probabilistic structure of the dataset. The focus was on techniques that model this distribution explicitly, offering insights into how data behaves and can be represented mathematically.

I explored latent variable models, particularly their role in learning compressed, interpretable representations of data. These models include Autoencoders, Variational Autoencoders (VAEs), and Generative Adversarial Networks (GANs). Autoencoders, an unsupervised approach, learn lower-dimensional representations by compressing data into a latent vector and reconstructing the input data using a decoder. A key component is the bottleneck hidden layer, which forces the network to capture compact and meaningful latent representations. However, due to their deterministic nature, Autoencoders often reconstruct the same output for the same input. To address this limitation, VAEs introduce stochasticity and probabilistic priors, such as a Gaussian distribution, to improve latent space learning. VAEs regularize the encoding process, ensuring latent representations are evenly distributed and discouraging clustering in specific regions. This regularization introduces both continuity (similar latent points decode into similar outputs) and completeness (samples from latent space produce meaningful outputs). Additionally, latent perturbation, where individual latent variables are varied while keeping others fixed, highlighted how different latent dimensions encode specific, interpretable features.

Moving to Generative Adversarial Networks (GANs), I studied their ability to generate realistic samples by directly transforming low-dimensional noise vectors into complex data representations. GANs consist of two competing networks: a Generator, which attempts to create data that mimics the real dataset, and a Discriminator, which differentiates between real and fake data. Unlike models that explicitly model data density, GANs rely on sampling and transformations to match the data distribution. Training involves a competition between the Generator and Discriminator, gradually improving the Generator’s ability to produce realistic samples. GANs stand in contrast to diffusion models, which generate samples iteratively by refining noise over time.

## Multi-modal Learning

Lastly, I explored multi-modal learning, an area focused on integrating data from different modalities such as text, images, audio, and video. Multi-modal models rely on specialized embeddings and fusion techniques to create unified data representations that capture relationships across modalities. Each modality undergoes preprocessing to generate embeddings—examples include Word2Vec for text, CNNs for images, and numerical representations for structured data. These embeddings are then fused to produce a final multi-modal representation. Fusion can occur at different stages: early fusion, where modalities are combined early in the pipeline; intermediate fusion, which concatenates feature representations before prediction; and late fusion, which combines outputs of independently processed modalities. State-of-the-art architectures like Data2Vec and VilBERT demonstrate the potential of these techniques in handling multi-modal data effectively. This week’s learning emphasized the importance of embedding generation and fusion strategies in enabling AI models to understand and process complex, multi-modal inputs.

# Report 3

Describe what you have learned during this study period. You can also include references to external material you have used to enhance your learning process (e.g., book chapters, scientific papers, online courses, online sites).

| Task     | Time Spent (hours) |
| -------- | ------------------ |
| Lectures | 6                  |
| Labs     | 12                 |

## Stable Diffusion

This week, I explored the concepts of Variational Autoencoders (VAEs) and U-Net architectures, focusing on their applications in image processing, alongside an understanding of Stable Diffusion. A VAE, incorporating a U-Net with a cross-attention mechanism, transforms images from pixel space to a latent representation through its encoder, downsampling to reduce complexity. The U-Net within the VAE denoises the latent representation iteratively, reversing diffusion steps to reconstruct a sharp image via the decoder. U-Net, named for its U-shaped structure, is designed for image segmentation. Its encoder extracts high-level semantic features by reducing spatial dimensions through repeated convolutional blocks, max pooling, and doubling the number of channels after each downsampling. At the bottleneck, the feature map has minimal spatial dimensions but rich semantic content due to numerous channels. The decoder restores spatial resolution using upsampling, convolutional layers to refine features, and halving the number of channels per upsampling, producing a pixel-wise output while preserving semantic information. Skip connections between the encoder and decoder enhance feature integration. Stable Diffusion, a related technique, leverages a similar diffusion-based approach, using a U-Net to iteratively denoise a latent representation starting from random noise, guided by a text prompt through cross-attention mechanisms to generate high-quality images. Unlike traditional VAEs, Stable Diffusion operates in latent space for efficiency and is trained on large datasets to produce diverse, photorealistic outputs, making it a powerful tool for text-to-image generation.

## Project

I have also dedicated a fair amount of time to work on the group project on car detection in snow.

# Report 4

Describe what you have learned during this study period. You can also include references to external material you have used to enhance your learning process (e.g., book chapters, scientific papers, online courses, online sites).

| Task     | Time Spent (hours) |
| -------- | ------------------ |
| Lectures | 6                  |
| Labs     | 12                 |
| Project  | 10                 |

## Generation Techniques

In exploring Google DeepDream, I’ve gained a deeper understanding of how neural networks perceive and amplify patterns in visual data. What stands out to me is how this program deviates from traditional optimization approaches by not relying on a predefined loss function. Instead, it focuses on amplifying features already recognized by the CNN. By selecting specific layers or neurons, the input image is iteratively modified to emphasize these features, creating a fascinating feedback loop. This process doesn’t aim to correct errors but rather to highlight and exaggerate the patterns the network identifies. The surreal and dreamlike visuals produced by DeepDream reveal the intricate and often unexpected ways neural networks interpret images. It’s fascinating to see how the inner workings of a CNN can be made visible, providing both artistic and analytical insights.

Learning about style transfer has been equally captivating, as it combines the structural essence of one image with the artistic flair of another. I’ve come to appreciate how CNNs extract different elements of an image through their layers—deeper layers focusing on high-level content and structure, and shallower layers capturing textures, colors, and artistic patterns. The balance between content loss and style loss is particularly intriguing. Content loss ensures that the output retains the original image’s structure, while style loss imbues it with the textures and colors of the chosen style. This technique demonstrates the incredible versatility of neural networks, as they can transform simple photographs into masterpieces inspired by famous artists. It’s a powerful reminder of how art and technology intersect, opening up endless possibilities for creative expression.

## Project

I've have also spent a fair amount of time training a few YOLO models to establish some basic comparisons.

# Report 5

Describe what you have learned during this study period. You can also include references to external material you have used to enhance your learning process (e.g., book chapters, scientific papers, online courses, online sites).

| Task     | Time Spent (hours) |
| -------- | ------------------ |
| Lectures | 5                  |
| Labs     | 10                 |
| Project  | 8                  |

## Dimensions

This week, I explored the concept of dimensions in datasets and the challenges associated with high-dimensional data, particularly the phenomenon known as the curse of dimensionality. In data science, a dimension represents a feature or variable, such as speed, weight, or fuel efficiency in a dataset about cars. High-dimensional datasets, while rich in information, introduce complexities due to the curse of dimensionality. This phenomenon has advantages, like capturing more detailed information about data points, which can lead to more nuanced models. However, as dimensions increase, data points become sparsely distributed, making it harder for machine learning models to find meaningful patterns. This sparsity can lead to overfitting, increased computational costs, and poor model performance.

## Dimensionality Reduction

To address these challenges, I learned about two powerful dimensionality reduction techniques: Principal Component Analysis (PCA) and t-SNE.

I started with PCA, a technique designed to reduce the number of features in a dataset while preserving as much of the data’s variation as possible. PCA identifies principal components, which are new directions in the data where variance is maximized. The first principal component (PC1) captures the direction with the highest variance, and the second principal component (PC2) captures the next highest, but it must be orthogonal (perpendicular) and uncorrelated to PC1. This orthogonality ensures each component provides unique information, helping to combat the curse of dimensionality by reducing the number of features. Suppose a dataset of 10 students, each with two features: hours studied and hours slept before an exam. These features are correlated (more study time might mean less sleep). PCA might find that PC1 represents a combination of study and sleep hours, capturing most of the variation, while PC2 represents the difference between study and sleep hours. If PC1 explains 90% of the variance, one can reduce the dataset to just PC1, simplifying it from two dimensions to one while retaining most of the information. This makes the data less sparse and easier to model.

Next, I explored t-SNE, a technique primarily used for visualizing high-dimensional data in 2D or 3D. Unlike PCA, t-SNE focuses on preserving the local structure of the data, ensuring that points close in the high-dimensional space remain close in the lower-dimensional visualization. It does this by converting distances between points into probabilities representing similarity. Imagine a dataset of 20 fruit images (apples, oranges, and bananas), each described by 1,000 pixel values. In this high-dimensional space, the curse of dimensionality makes it hard to see patterns. t-SNE could reduce the data to a 2D scatter plot, where images of apples cluster together, oranges form another cluster, and bananas a third.

PCA is ideal for simplifying data for analysis or modeling by reducing the number of dimensions, making models more efficient and less prone to overfitting. t-SNE, on the other hand, excels at visualization, helping to uncover patterns in complex datasets.

## Project

I've have also spent more time exploring other kinds of detection models such as RF-DETR, Faster-RCNN and augmentation techniques.

# Report 6

Describe what you have learned during this study period. You can also include references to external material you have used to enhance your learning process (e.g., book chapters, scientific papers, online courses, online sites).

| Task     | Time Spent (hours) |
| -------- | ------------------ |
| Lectures | 4                  |
| Labs     | 10                 |
| Project  | 5                  |

## Reinforcement Learning

I’ve come to appreciate how different it is from other forms of machine learning like supervised or unsupervised learning. Unlike those methods, RL doesn’t rely on labeled datasets or clustering algorithms. Instead, it thrives on a reward signal, which acts as a delayed form of feedback, guiding the agent’s behavior over time. What’s fascinating is how time and causality play a crucial role here—each action not only affects the current outcome but also shapes the data the agent receives in the future.

The concept of maximizing cumulative rewards resonates deeply with me. It frames the agent’s purpose in a simple yet profound way: every decision made should ultimately work toward a long-term goal. The reward hypothesis, which suggests all objectives can be boiled down to this maximization, ties everything together in an elegant framework.

Understanding Markov states was another eye-opener. I learned that a state is Markovian if it contains all the necessary information from the past to predict the future, making the process inherently efficient. This Markov property simplifies complex systems by focusing only on the present state without needing to carry the weight of the entire history.

I also gained insights into the challenges agents face in environments that are not fully observable. In such cases, the agent must navigate uncertainties and infer the best course of action from incomplete data, adding an intriguing layer of complexity.

The major components of an RL agent—policy, value function, and model—offer a structured way to think about decision-making. I found it especially interesting how policies can be deterministic or stochastic, depending on whether actions are fixed or probabilistic. Additionally, value functions and models emphasize foresight, teaching agents to not just act but also predict and adapt.

Finally, exploring different types of RL agents, such as value-based, policy-based, and actor-critic, broadened my understanding of how diverse strategies can be employed depending on the problem at hand. Each approach comes with its strengths and trade-offs, reflecting the richness of RL as a field.

## TinyML

I’ve also gained a deeper appreciation for how machine learning can be adapted to fit the constraints of small, low-power devices. TinyML is not just a technological advancement but also a demonstration of how AI can be democratized, making powerful capabilities accessible on devices with limited resources. It has been fascinating to see how these tiny models enable real-world applications like smart doorbells and fitness trackers, blending intelligence with practicality in everyday tools.

## Project

I've have dedicated most of my time to look at various inference techniques that can help to detect small objects, SAHI in particular.
