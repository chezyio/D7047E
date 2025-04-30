# D7047E Advanced Deep Learning

Taken at Luleå University of Technology 🇸🇪 and mapped to SC4001 Neural Networks & Deep Learning 🇸🇬

## Topics

-   Convolutional Neural Networks (CNN)
    -   U-Net
    -   Fast-RCNN/Faster-RCNN
-   Recurrent Neural Networks (RNN)
    -   Long Short Term Memory (LSTM)
    -   Gated Recurrent Units (GRU)
-   Transformers
    -   Attention Mechanism
    -   Bidirectional Encoding from Transformers (BERT)
    -   Vision Transformers (ViT)
-   Generative Adversarial Networks (GAN)
-   Variational Autoencoder (VAE)
-   Stable Diffusion
-   Visual Quesion Answering (VQA)
-   Reinforcement Learning (RL)
-   Multi-modal Learning
-   Research and others
    -   Google DeepDream
    -   Google LeNet
    -   ResNet
    -   Super-resolution
    -   Style Transfer
    -   Dimensionality Reduction
        -   Principal Component Analysis (PCA)
        -   t-SNE
    -   Nautral Lanugage Processing (NLP)
        -   Distributional Hypothesis
-   Advanced Topics
    -   Artificial Curiosity
    -   AutoML
    -   Meta Learning
    -   Explainable AI
    -   Ethics and Bias

## Convolutional Neural Networks (CNN)

-   Convolution is the operation of applying a kernel/filter to input data to extract features
-   Convolution produces feature maps highlighting patterns
-   Feature maps represents detected features at different spatial locations
-   Filter/kernel is a small matrix (usually 3x3 or 5x5) that slides over input to compute feature maps
    -   Each filter/kernel detects a specifc pattern
-   Channels represent the depth of a feature map where each channel encodes different features
    -   Channels are essentially a stack of feature maps
-   Spatial dimensions represent the height and width of the an image or feature map
-   Pooling downsamples image, reducing spatial dimensions
    -   Max pooling
    -   Average pooling
-   Downsampling reduces spatial dimensions and focuses on high-level features, loses fine spatial deails
-   Upsampling increases spatial dimensions and recovers resolution
    -   Bilinear interpolation
    -   Transposed convolution
-   Semantic information is captured in deeper layers which are abstract and context-focused
-   Spatial information captures precise location and structure of data and usually preserved in early layers

### U-Net

-   Developed for image segmentation
-   Named after its U-shaped structure of combining an encoder and a decoder with skip connections
-   Encoder extracts high-level semantic information by reducing spatial dimensions and increasing feature depth
    -   Repeated blocks of convolution to extract features
    -   Max pooling for downsampling
    -   Number of channels doubles after each downsampling
-   At the bottom of the U shape lies the bottleneck where the feature map has small spatial dimensions but many channels with rich semantic features
-   Decoder recovers spatial resolution to produce a pixel-wise output while retaining semantic information
    -   Upsampling to double spatial dimensions
    -   Convolutional layers to refine features
    -   Number of channels halves after each upsampling
-   Semantic vs spatial information
    -   Encoder captures semantics
    -   Decoder restores spatial information

### Fast-RCNN

-   Popular object detection frameworks that builds on R-CNN
-   Improves over R-CNN by processing entire image once with CNN, rather than cropping and resizing regions individually
-   Uses ROI pooling to extract fixed size feature maps for each region proposal from the shared CNN feature map
-   Combines feature extraction, region classification and bounding box regression into a single network

### Faster-RCNN

-   Incorporates a Region Proposal Network (RPN) the eliminates the dependency on external region proposal algorithms
    -   RPN generates region proposals directly from feature maps of CNN

## Recurrent Neural Networks (RNN)

![title](./assets/LSTM3-SimpleRNN.png)

-   Networks with loops in them, allowing information to persist
-   Uses a tanh function to regulate outputs between -1 and 1
-   Trained on sequential or time series data that can make sequential predictions
-   Can't learn handle long-term dependencies well enough becuase of vanishing gradients early in the network

## Long Short Term Memory (LSTM)

![title](./assets/LSTM3-chain.png)

-   Special kind of RNN
-   Learning long-term depedencies is its default behaviour
-   LSTM have the ability to remove or add information to cell state using gates
-   Cell state is a long-term memory component that stores information over long time steps
    -   Updated by forget and input gates
-   Hidden state is a short-term memory component that holds information relevant to the current time step and is used for predictions to the next time step
    -   Dervied from cell state and output gate
-   Each LSTM cell consists of
    -   Forget gate
    -   Input gate
    -   Output gate

### Forget Gate

-   Decide what information to throw away from the cell state
-   Decision is made using a sigmoid layer
    -   Outputs a number between 0 and 1
    -   0 means "completely forget this"
    -   1 means "completely keep this"

### Input Gate

-   Decide what new information to be added to cell state
-   Sigmoid layer decides whicih values to update
-   Tanh layer creates a vector of new candidate values that should be added to cell state
-   Combine sigmoid and tanh layers to update state

### Updating old cell state $C_{t-1}$ into new cell state $C_t$

-   Multiply old cell state by $f_t$, forgetting the things decided earlier
-   Then add $i_t * \tilde{C_t}$
-   **This is where information is being dropped and added**

### Output Gate

-   Output will be based on cell state but filtered version
-   First, run a sigmoid layer to decide what parts of cell state is going to output
-   Then put cell state through tanh and multiply it by output of sigmoid gate

## Gated Recurrent Units (GRU)

![title](./assets/LSTM3-var-GRU.png)

-   Dramatic Variation of LSTM
-   Combines forget and input gate into a single "update" gate
-   "update" gate decides what information to discard and add
-   "reset" gate decides how much past information to forget
-   No cell state, instead use hidden state to transfer information
-   Arguably more efficient that LSTM

## Transformers

-   Used for sequence to sequence tasks like NLP
-   Relies entirely on attention mechanism to model relationship between elements in a sequence
-   Uses a encoder-decoder architecture
    -   Encoder processes input sequence into a contextualized representation using stacked layers of multi-head attention and feed-forward networks
    -   Decoder generates output sequences (autoregressive in nature) by attending to encoder outputs and its own partial outputs
-   Attention mechanism
    -   Self-attention computes relationships between all pairs of input element, assigning weights to indicate importance
    -   Scaled dot-product is the core operation calculating query-key dot products, scaled and softmax-normalized to weigh value vectors
    -   Multi-head attention runs multiple attention operations in parallel, capturing diverse relationships then concatenates outputs
-   Positional encoding adds fixed or learned vectors to input embeddings to encode positions of elements
-   Input tokens are being converted to dense vectors, often augmented with positional encoding
-   Steps

    1. Sequence of text is converted into embedings
    2. Encoder applies self-attention to model relationships, followed by feed-forward network
    3. Decoder layer use masked self-attention (for autoregressive generation) and cross-attention (to encoder outputs)

-   Unlike RNNs, transforners are highly parallelizable and can process all elements simultaeneously

### Attention

-   Query (Q), Key (K) and Value (V) vectors are the core components of the attention mechanism, specifically the scaled dot-product attention
-   Q represents a question of request for infornation about relevance of other elements in the sequence
    -   "Query" how much attention to pay to each element
-   K acts as a label or identifier for each element, used to match against Q to determine relevance
-   V contains the actual content or information of each element which is weighted and aggregated based on attention scores
-   Q, K and V are derived from input embeddings through learnable linear transformations
-   Steps

    1. Start with a sequence of input tokens, each represneted as an embedding of dimension $d_{model}$
    2. For a sequence of length $n$, the input is a matrix $X \in \R^{n \times d_{model}}$
    3. For each token, vectors Q, K and V are computed

        3.1. $Q=XW_Q$, $K=XW_K$, $V=XW_V$ where $W_Q$, $W_K$, $W_V$ are learned weight matrices

        3.2. $d_k$ is the dimension of Q, K, V and $d_k = d_v = d_{model}/h$, where $h$ is the number of attention heads (e.g., 64 for 8 heads would give $d_{model}=512$)

    4. To calculate the attention scores, we need to compute the similarity between Q and K

        4.1. $\text{Similarity} = QK^T \in \R^{n \times n}$

        4.2. $\text{Scaled Scores} = \frac{QK^T}{\sqrt{d_k}}$, $\sqrt{q_k}$ is used for numerical stability to prevent large values that can potentially destabilize gradients

    5. To calculate the attention weights, apply softmax to scaled scores

        5.1. $\text{Attention Weights} = \text{softmax}(\frac{QK^T}{\sqrt{d_k}}) \in \R^{n \times n}$

    6. To get weighted sum of values, multiply attention weights by value vectors get final output

        6.1. $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V \in \R^{n \times d_v}$

        6.2. Each output vector for a token $i$ is a weighed combination of all value vectors, emphasizing tokens with high attention scores

### Multi-head Attention

-   The use of multiple heads allows us to capture diverse relationships
-   Split $Q, K, V$ into $h$ smaller heads, each with dimension $d_k/h$
-   Attention is independently computed for each head and results are concatenated at the end before passing through a linear layer

### Toy Example

Input Sequence: 3 tokens ("the brown fox"), each with a 4-dimensional embedding.

Input Matrix ( X ):

```math
X = \begin{bmatrix} 1 & 0 & 0 & 1 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix} \in \mathbb{R}^{3 \times 4}
```

Sequence length: $n = 3$

Embedding dimension: $d_{\text{model}} = 4$

Dimension for $Q$, $K$, $V$: $d_k = d_v = 2$ (for single-head and per head in multi-head)

<u>Single-head Attention</u>

Weight matrices:

```math
W_Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}, \quad W_K = \begin{bmatrix} 0 & 1 \\ 1 & 0 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}, \quad W_V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \\ 0 & 0 \end{bmatrix} \in \mathbb{R}^{4 \times 2}
```

Step 1: Compute $Q$, $K$, $V$

```math
Q = X W_Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix}, \quad K = X W_K = \begin{bmatrix} 0 & 1 \\ 1 & 0 \\ 0 & 0 \end{bmatrix}, \quad V = X W_V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix} \in \mathbb{R}^{3 \times 2}
```

Step 2: Attention Scores

```math
Q K^T = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \end{bmatrix} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}
```

Scale by $\sqrt{d_k} = \sqrt{2} \approx 1.414$:

```math
\frac{Q K^T}{\sqrt{d_k}} = \begin{bmatrix} 0 & 0.707 & 0 \\ 0.707 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}
```

Step 3: Attention Weights

Apply softmax per row:

-   Row 1: $\text{softmax}([0, 0.707, 0]) \approx [0.244, 0.512, 0.244]$
-   Row 2: $\text{softmax}([0.707, 0, 0]) \approx [0.512, 0.244, 0.244]$
-   Row 3: $\text{softmax}([0, 0, 0]) \approx [0.333, 0.333, 0.333]$

```math
\text{Attention Weights} = \begin{bmatrix} 0.244 & 0.512 & 0.244 \\ 0.512 & 0.244 & 0.244 \\ 0.333 & 0.333 & 0.333 \end{bmatrix}
```

Step 4: Attention Output

```math
\text{Attention}(Q, K, V) = \text{Attention Weights} \cdot V = \begin{bmatrix} 0.244 & 0.512 & 0.244 \\ 0.512 & 0.244 & 0.244 \\ 0.333 & 0.333 & 0.333 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 0.244 & 0.512 \\ 0.512 & 0.244 \\ 0.333 & 0.333 \end{bmatrix}
```

Token 1: $[0.244, 0.512]$, leans toward Token 2's value $[0, 1]$ (weight 0.512).

Token 2: $[0.512, 0.244]$, leans toward Token 1's value $[1, 0]$ (weight 0.512).

Token 3: $[0.333, 0.333]$, equal mix due to uniform weights.

<u>Multi-head Attention</u>

Number of heads: $h = 2$

Per head: $d_k = d_v = d_{\text{model}} / h = 4 / 2 = 2$

Output projection: $W_O \in \mathbb{R}^{4 \times 4}$

Head 1

Use same weight matrices as single-head:

```math
W_Q^1 = W_Q, \quad W_K^1 = W_K, \quad W_V^1 = W_V
```

Q, K, V: Same as single-head

Attention Weights: Same as single-head

Head 1 Output:

```math
\text{head}_1 = \begin{bmatrix} 0.244 & 0.512 \\ 0.512 & 0.244 \\ 0.333 & 0.333 \end{bmatrix}
```

Head 2

New weight matrices:

```math
W_Q^2 = \begin{bmatrix} 0 & 1 \\ 1 & 0 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}, \quad W_K^2 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}, \quad W_V^2 = \begin{bmatrix} 0 & 1 \\ 1 & 0 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}
```

Compute Q, K, V:

```math
Q^2 = X W_Q^2 = \begin{bmatrix} 0 & 1 \\ 1 & 0 \\ 0 & 0 \end{bmatrix}, \quad K^2 = X W_K^2 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix}, \quad V^2 = X W_V^2 = \begin{bmatrix} 0 & 1 \\ 1 & 0 \\ 0 & 0 \end{bmatrix}
```

Attention Scores:

```math
Q^2 K^{2T} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}, \quad \frac{Q^2 K^{2T}}{\sqrt{d_k}} = \begin{bmatrix} 0 & 0.707 & 0 \\ 0.707 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}
```

Attention Weights: Same as Head 1 (due to symmetric design):

```math
\text{Attention Weights}^2 = \begin{bmatrix} 0.244 & 0.512 & 0.244 \\ 0.512 & 0.244 & 0.244 \\ 0.333 & 0.333 & 0.333 \end{bmatrix}
```

Head 2 Output:

```math
\text{head}_2 = \text{Attention Weights}^2 \cdot V^2 = \begin{bmatrix} 0.244 & 0.512 & 0.244 \\ 0.512 & 0.244 & 0.244 \\ 0.333 & 0.333 & 0.333 \end{bmatrix} \begin{bmatrix} 0 & 1 \\ 1 & 0 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 0.512 & 0.244 \\ 0.244 & 0.512 \\ 0.333 & 0.333 \end{bmatrix}
```

Concatenate Head Outputs

```math
\text{Concat}(\text{head}_1, \text{head}_2) = \begin{bmatrix} 0.244 & 0.512 & 0.512 & 0.244 \\ 0.512 & 0.244 & 0.244 & 0.512 \\ 0.333 & 0.333 & 0.333 & 0.333 \end{bmatrix} \in \mathbb{R}^{3 \times 4}
```

Final Projection

Use identity matrix for simplicity:

```math
W_O = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}
```

```math
\text{MultiHead Output} = \text{Concat}(\text{head}_1, \text{head}_2) \cdot W_O = \begin{bmatrix} 0.244 & 0.512 & 0.512 & 0.244 \\ 0.512 & 0.244 & 0.244 & 0.512 \\ 0.333 & 0.333 & 0.333 & 0.333 \end{bmatrix}
```

Interpretation

Token 1: $[0.244, 0.512, 0.512, 0.244]$, combines Head 1's focus on Token 2's $[0, 1]$ and Head 2's focus on Token 2's $[1, 0]$.

Token 2: $[0.512, 0.244, 0.244, 0.512]$, combines Head 1's focus on Token 1's $[1, 0]$ and Head 2's focus on Token 1's $[0, 1]$.

Token 3: $[0.333, 0.333, 0.333, 0.333]$, neutral in both heads.

### Bidirectional Encoder Representations from Transforner (BERT)

-   Learns to represent text as a sequence of vectors using self-supervised
-   Uses encoder-only from transfomer architecture

## Vision Transformer (ViT)

## Generative Adversarial Networks (GAN)

-   Idea is to use 2 neural networks to compete with each other
    -   Generator turns noise into an imitation of data to try trick the discriminator
    -   Discriminator tries to identify real data from fakes created by generator
-   2 models are trained simultaeneously in a zero-sum game where the generator improves by trying to trick the discriminator and the discriminator improves by getting better at spotting fakes
-   Process continues until Nash Equilibrium is reached where generator outputs are so realistic that discriminatot can't differentiate
-   For the generator
    -   A random noise vector $z$ is drawn from a Gaussian distribution as input
    -   Outputs $G(z)$ that ideally resembles real data
-   For the discriminator
    -   Either real data $x$ or fake data $G(z)$ can be taken as an input
    -   A scalar probability $D(x)$ or $D(G(z))$ between 0 (fake) and 1 (real) os the output
-   Training procsss is a minimax game

    -   Discriminator's goal is to maximise probability of correctly classifying real dataa and fake data
    -   Generator's goal is to minimize probability of discriminator's output being correctly identifying as fake

-   Training steps

    -   Update Discriminator:

        -   Sample real images from the dataset (e.g., photos of cats).
        -   Sample noise $z$ and generate fake images using the generator.
        -   Compute the discriminator’s loss:
            -   Reward it for labeling real images as real $D(\text{real}) \approx 1$.
            -   Reward it for labeling fake images as fake $D(\text{fake}) \approx 0$.
        -   Update the discriminator’s weights via backpropagation.

    -   Update Generator:

        -   Sample noise $z$ and generate fake images.
        -   Pass fake images through the discriminator to get $D(\text{fake})$.
        -   Compute the generator’s loss based on how well it “fooled” the discriminator (i.e., maximize $D(\text{fake})$.
        -   Update the generator’s weights via backpropagation.

    -   Repeat:
        -   Alternate these steps for thousands of iterations. Typically, the discriminator is updated more frequently (e.g., 5 times per generator update) because it learns faster.

-   Mode collapse can happen when generator produces limited varieties of outputs, ignoring parts of the real data distribution
-   Vanishing gradients can occur if the discriminator becomes too good too quickly where it assigns near-zero probabilities to fake data, leaving generator with little gradient to learn from

## Stable Diffusion

-   Model consists of VAE with U-Net based on a cross-attention mechanism to handle various input modalities
-   Encoder block of VAE transform image from pixel space to latent representation, downsampling image to reduce complexity
-   Image is denoised using U-Net iteratively to reverse diffusion steps and reconstruct sharp image using VAE decoder block

## Autoencoders

-   Unsupervised approach for learning a lower-dimensional feature representation from unlabeled training data
-   Lower diemnsional space would effectively mean compressing into a small latent vector and leanring a very compact and rich feature representation
-   The latent space can be learnt by training to model to use the features to reconstruct the original data, this is done by using a decoder
-   Afterwards, train the network by comparing the orignal input and generated output and minimizing the loss
-   Bottleneck hidden layer forces the network to learn a compressed latent representation
-   Reconstruction loss forces the latent representation to capture or encode as much information as possible
-   Since layer Z is deterministic, it will always reconstruct the same output given the same weight
-   Stochasticity needs to be introduced to learn the latent space well, this is where VAE comes into play

## Variational Autoencoders

-   Builds on autoencoders
-   VAEs are a probabilistic twist on autoenconders
-   Stochasticity needs to be introduced to learn the latent space well, this is where VAE comes into play

## Visual Question Answering (VQA)

-   Involves teaching computers to connect the dots between images and language
-   Given an image of a park and question of "how many trees are there"
    -   Computer needs to analyze the image, identify and count the trees — CV
    -   Computer also needs to comprehend and respond in a human-like manner — NLP
-   General methodology
    -   Extract features from question
    -   Extract features from image
    -   Combine features to generate answer
-   Feature extraction
    -   Bag-of-Words (BOW) or LSTM can be used for text
    -   CNNs are typically used for images
-   Combine features
    -   Straightforward combination using concatenation followed by input into linear classifier
    -   Use bayesian models to deduce inherent relationships between feature distributions of image, question and answer

## Multi-modal Learning

-   Multi-modal model requires specialized embeddings and fusion modules to create representations of the different modalities
-   Multi-modal deep learning trains AI models that combine information from several types of data simultaeneously to learn their unified data representations
-   Modern AI architectures can learn cross-modal relationships and semantics from different data types
-   Data types include text, image, audio, video and more
-   Raw data needs to be transformed into format understood by model
    -   Numerical data can be fed directly
    -   Text must be converted into word embeddings
    -   Images must be converted to pixels
-   These various modalities have to be individually processed to generate embeddings and then fused
-   Final representation is amalgamation of the information from all data modalities
-   Unimodal encoders like Word2Vec for natural language processing tasks and CNN to encode images are the traditional way of generating data embeddings
-   SOTA architectures like Data2Vec, VilBERT can handle multiple modalities
-   After feature extraction, next step is to multi-modal fusion
    -   Early — combines data from various modalities early on in training pipeline
    -   Intermediate — or feature-level fusion, concatenates feature representations from each modality before making predictions
    -   Late — processes each modality through model independently and returns individual outputs, independent predictions are then fused at later stage using averaging

## Reinfocement Learning (RL)

-   When compared to supervised and unsupervised learning, RL does not have a supervisor, only a reward signal
    -   Feedback is delayed and not instantaneous
    -   Time matters
    -   Agent's actions affect the subsequent data it receives
-   At each step $t$, the agent
    -   Executes action $A_t$
    -   Receives observation $O_t$
    -   Receives scalar reward $R_t$

### Reward

-   A reward ${R_t}$ is a scalar feedback signal
-   Used to indicate how well agent is doing at step $t$
-   Agent's job is to maximise the cumulative reward
-   Reward hypothesis states that all goals can be described by the maximisation of expected cumulatuve reward

### Information State

-   An information state, also known as Markov state contains all useful information from the history
-   A state $S_t$ is Markov if and only if $\mathbb{P}[S_{t+1}|S_t] = \mathbb{P}[S_{t+1}|S_1,...,S_t]$
    -   Markovian propety states that transition properties depend only on current state and not on previous history

### Observable Environments

-   Full observability would mean the agent directly observes the environment state ($O_t = S^a_t = S^e_t$)
-   Partially observable environemnts would mean agent indirectly observe environment and agent state does not eqaute to environment state

### Major Components of RL agent

-   Policy is action that an agent takes in any given state
    -   Map from state to action
    -   Can be deterministic or stochastic
-   Value function defines how good is each state and/or action
    -   Predict future reward
-   Model is the agent's representation of the environment
    -   Predicts what the environment will do next
    -

### Types of RL agent

-   Value-based
-   Policy-based
-   Actor Critic

## Research and Others

### Google DeepDream

-   DeepDream is a computer vision program created by Google to visualize and understand the features learned by CNNs
-   Does not have a predefined loss function, instead it focuses on amplifying features already recognized by the model
-   Process involves selecting specific layers or neurons in the network and modifying the input image to amplify their activations
-   Results in feedback loops that emphasize patterns the network see in the image rather then correcting for any errors

### Style Transfer

-   Technique in computer vision that involes merging content of one image with the artistic style of another
-   Leverages CNNs to create visually stunning images that look like a photograph painted in the style of famous artists
-   Content can be extracted from deeper layers of the CNN capture high-level structure and sematic information while style can be extracted from the shallow and mid-layers of a CNN that encodes patterns such as textures, colors and strokes
-   Typically has 2 loss functions
    -   Content loss ensures output image retains high-level structure of content
    -   Style loss ensures the output captures the texture and color patterns

### Principal Component Analysis (PCA)

-   PCA is a dimensionality reduction technique used to simplify datasets by reducing the number of features while preservering as much variance as possible
-   PCA identifies the principal components along which data varies the most
    -   PC1 typically denotes the component with the highest variance
    -   PC2 typically denotes the component with the second highest variance
    -   PC1 and PC2 must be uncorrelated and orthogonal to each other

### t-SNE

-   t-SNE is a dimensionality reduction technique primarily used for visualizing high-dimensional data in 2D or 3D space
-   Focuses on preserving local structure where points that are close in high-dimensional space remain close in the lower-dimensional representation
-   Converts distances between data points into probabilities for similarity

### Nautral Lanugage Processing (NLP)

#### Distributional Hypothesis

-   "Words that occur in similar context tend to have similar meanings"
-   Underpinds much of modern computational linguistics and many word representation techniques
-   The semantics of a word can be inferred from the contexts in which it appears
    -   "cat" and "dog" often occur in similar context
-   Words that co-occur frequently or appear in similar surrounding words are asuumed to share semantic properties
    -   "king" and "queen" often appear with terms like "royalty", "palace" and "throne"
-   In short, the meaning of word is represented in a vector in high-dimensional space based on its distributional properties in a corpus
-   Word2Vec is a popular technique for generating dense, continuous vector representations of words, capturing semantics and syntactic relationships based on contexts where they appear
    -   Can be trained using Skip-Gram and Continuous Bag of Words (CBOW)

#### Skip-gram

-   Model used to predict the surrounding words of a target word in a sentence
-   Given word "dog", it predicts "the", "bark" and "loudly"

#### Continuous Bag of Words (CBOW)

-   Predicts the target word based on its surrounding context words
-   Given context "the", "bark" and "loudly", it predicts "dog"

### TinyML

-   A way to run ML models on small, low-power devices like microcontrollers
-   Common applications include voice assisstants on devices, smart doorbells and fitness trackers
-   Primarily designed to bring AI capabilities to devices that don't have much processing power, memory, battery life or limited internet connectivity
-   Uses small, optimized models that are compressed to fit on tiny devices
-   Training is done on powerful computers while inference is done on small device

#### Building TinyML Models

-   Pruning, knowledge distillation and quantization are key techniques for optimizing ML models to run on resource-constrained devices
-   Pruning involves removing parts of a neural netwwork that contribute little to its predictions, making the model smaller and faster
    -   Similar to the concept of trimming unnecessary branches from a tree
    -   Analyze model to find weights with low impact on the output
    -   Retrain the model after pruning
-   Types of pruning
    -   Weight pruning sets small weights to zero, creating a spare model
    -   Neuron pruning removes the entire neuron itself
-   Over-pruning can hurt accuracy, so must balance size reduction and performance

#### Knowledge Distillation

-   Adopts the concept of teacher-student training, where the student learns to miimic the teacher's behaviour while being lightweight
-   A technique where a large, accurate model (the teacher) transfers its knowledge to a smaller, simpler model (the student)
-   The student model can be trained to match both the true labels and the teacher's soft prediction, allowing it to learn nuanced patterns

#### Quantization

-   Quantization reduces the precision of a model's weights and computations, making it smaller and faster
-   Instead of using 32-bit FP, quantization uses 8-bit integers that takes up lesser memory and making it faster
-   Ways to quantize
    -   Post-training Quantization (PTQ) — after training a model with FP weights, convert it to use inteeers
    -   Quantization-aware Training (QAT) — incorporates quantization effects during training, allowing the model to adapt to low-precision constraints, whereas PTQ applies quantization as a post-processing step
-   Types of quantization
    -   Weight-only — only weights are quantized
    -   Full quantization — both weights and activations are quantized, reducing size and speeding up inference
    -   Dyanmic range quantization — quantizes weights statically but activations dynamically during inference
    -   Integer-only quantization — ensures all operations use integers

### Artificial Curiosity

-   Design of AI systems that mimic human-like curiosity to explore, learn and adapt in dynamic environments
-   Unlike tradtional AI that relies on extrinsic rewards (e.g., acheiving a specific goal), artificial curiosity is driven by intrinsic motivation where the AI explores because it wants to understand or discover something new
    -   Intrinsic rewards include novelty (encountering new states), surprise (unexpected outcome), or prediction error (gaps between expected and actual result)
-   Balances exploration with exploitation
-   Curiosity onften involes maximizing information gain, where AI seeks to reduce uncertainty about environment
    -   Can be modelled using Bayesian surprise

### AutoML

-   Automation of end-to-end process of applying ML to real-world problems
-   Aims to make ML accessible to non-experts by automating task like data preprocessing, feature engineering, model selection, hyperparameter tuning and deployment
-   Common techniques include
    -   Bayesian optimization — models the performance of hyperparameters to efficiently search for optimal settings
    -   Genetic algorithms — evolves model architecture or pipelines over generations
    -   RL — guides search for optimal models or architectures
    -   Gradient-based methods — optimizes neural architectures directly

### Meta-learning

-   A paradigm where model learns how to learn new tasks efficiently by leveraging experience from related tasks
-   Goal is to create AI that generalizes quickly to new problems with minimal data or training
-   Meta-learning assumes a distribution of tasks where the model learns a general strategy to adapt to new tasks from this distribution
-   Few-shot learning is commonly used to learn a new task with very few samples of data
-   Types of meta-learning
    -   Metric-based — learns a similarity metric to compare new exmaples to known ones
    -   Model-based — learns a model that can quickly adapt
    -   Optimization-based — learns an initialization or optimization strategy that enables fast adaptation

## References

Understanding LSTM Networks. (2015, August 27). https://colah.github.io/posts/2015-08-Understanding-LSTMs/
