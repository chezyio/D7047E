# D7047E Advanced Deep Learning

Taken at Luleå University of Technology 🇸🇪 and mapped to SC4001 Neural Networks & Deep Learning 🇸🇬

## Topics

-   Convolutional Neural Networks (CNN)
-   Recurrent Neural Networks (RNN)
-   Long Short Term Memory (LSTM)
-   Transformers
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
TO BE UPDATED

#### Faster-RCNN
TO BE UPDATED

### Recurrent Neural Networks (RNN)

![title](LSTM3-SimpleRNN.png)

-   Networks with loops in them, allowing information to persist
-   Trained on sequential or time series data that can make sequential predictions
-   Can't learn handle long-term dependencies well enough

### Long Short Term Memory (LSTM)

![title](LSTM3-chain.png)

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

TO BE UPDATED

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
-   Given an image of a park and question of “how many trees are there”
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
