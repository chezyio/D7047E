# D7047E Advanced Deep Learning

Taken at Luleå University of Technology 🇸🇪 and mapped to SC4001 Neural Networks & Deep Learning 🇸🇬

## Topics

-   Convolutional Neural Networks (CNN)
    -   U-Net
    -   Fast-RCNN/Faster-RCNN
-   Recurrent Neural Networks (RNN)
    -   Long Short Term Memory (LSTM)
-   Transformers
    -   Bidirectional Encoding from Transformers (BERT)
    -   Vision Transformers (ViT)
-   Generative Adversarial Networks (GAN)
-   Variational Autoencoder (VAE)
-   Stable Diffusion
-   Visual Quesion Answering (VQA)
-   Reinforcement Learning (RL)
-   Multi-modal Learning

## Key Concepts

### Convolutional Neural Networks (CNN)

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

#### U-Net

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

#### Faster-RCNN

TO BE UPDATED

### Recurrent Neural Networks (RNN)

![title](./assets/LSTM3-SimpleRNN.png)

-   Networks with loops in them, allowing information to persist
-   Trained on sequential or time series data that can make sequential predictions
-   Can't learn handle long-term dependencies well enough

### Long Short Term Memory (LSTM)

![title](./assets/LSTM3-chain.png)

-   Special kind of RNN
-   Learning long-term depedencies is its default behaviour
-   LSTM have the ability to remove or add information to cell state using gates
-   Each LSTM cell consists of
    -   Forget gate
    -   Input gate
    -   Output gate

#### Forget Gate

-   Decide what information to throw away from the cell state
-   Decision is made using a sigmoid layer
    -   Outputs a number between 0 and 1
    -   0 means "completely forget this"
    -   1 means "completely keep this"

#### Input Gate

-   Decide what new information to be added to cell state
-   Sigmoid layer decides whicih values to update
-   Tanh layer creates a vector of new candidate values that should be added to cell state
-   Combine sigmoid and tanh layers to update state

#### Updating old cell state $C_{t-1}$ into new cell state $C_t$

-   Multiply old cell state by $f_t$, forgetting the things decided earlier
-   Then add new information ....
-   **This is where information is being dropped and added**

#### Output Gate

-   Output will be based on cell state but filtered version
-   First, run a sigmoid layer to decide what parts of cell state is going to output
-   Then put cell state through tanh and multiply it by output of sigmoid gate

### Transformers

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

#### Attention

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

#### Multi-head Attention

-   The use of multiple heads allows us to capture diverse relationships
-   Split $Q, K, V$ into $h$ smaller heads, each with dimension $d_k/h$
-   Attention is independently computed for each head and results are concatenated at the end before passing through a linear layer

#### Toy Example

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

### Vision Transformer (ViT)

### General Adversarial Networks (GAN)

-   GANs generate samples in one-shot directly from low-dimentional latent variables
-   Diffusion generate samples iteratively by repeatedly refining and remove noise
-   Idea is to use 2 neural networks to compete with each other
    -   Generator turns noise into an imitation of data to try trick the discriminator
    -   Discriminator tries to identify real data from fakes created by generator

### Autoencoders

-   Unsupervised approach for learning a lower-dimensional feature representation from unlabeled training data
-   Lower diemnsional space would effectively mean compressing into a small latent vector and leanring a very compact and rich feature representation
-   The latent space can be learnt by training to model to use the features to reconstruct the original data, this is done by using a decoder
-   Afterwards, train the network by comparing the orignal input and generated output and minimizing the loss
-   Bottleneck hidden layer forces the network to learn a compressed latent representation
-   Reconstruction loss forces the latent representation to capture or encode as much information as possible
-   Since layer Z is deterministic, it will always reconstruct the same output given the same weight
-   Stochasticity needs to be introduced to learn the latent space well, this is where VAE comes into play

### Variational Autoencoders

-   Builds on autoencoders
-   VAEs are a probabilistic twist on autoenconders
-   Stochasticity needs to be introduced to learn the latent space well, this is where VAE comes into play

### Visual Question Answering (VQA)

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

### Multi-modal Learning

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

### Stable Diffusion

-   Model consists of VAE with U-Net based on a cross-attention mechanism to handle various input modalities
-   Encoder block of VAE transform image from pixel space to latent representation, downsampling image to reduce complexity
-   Image is denoised using U-Net iteratively to reverse diffusion steps and reconstruct sharp image using VAE decoder block

## References

Understanding LSTM Networks. (2015, August 27). https://colah.github.io/posts/2015-08-Understanding-LSTMs/

skipgram
distributional hypotjhesis

Deep dream.
Style transfer

Applications of GAN for Image generation
general use cases
superresolution
Image to Image translation
Unpaired image translation
