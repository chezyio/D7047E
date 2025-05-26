# Exam Questions

## General Questions

1. What regularization techniques do you know?

    L1 regularization adds absolute values of the model's weights to the loss function by driving some weights to zero.

    L2 regularization adds squared values of model's weights to the loss function by penalizing large weights, encourage smaller and more evenly distributed weights.

    Dropout can be used to randomly deactivate neurons during training.

2. What is batch normalization and layer normalization?

    Batch normalization is a technique used in neural networks to stablilize and accelerate training by normalizing the inputs to each layer. It works by adjusting and scaling the activations with a mini-batch to have a mean of zero and a variance of one. Using batch normalization stabilizes training and allows the the network to converge faster.

    Layer normalization is a technique that normalizes the inputs of a layer across all features for each sampple independently, stabilizing and accelerating neural network training.

    Batch Normalization is typically used for CNN-based models with large, consistent batch sizes while Layer Normalization is typically used for transfomers, RNNs or when batch sizes are small.

3. What initialization methods do you know?

    Zero initialization sets all weights and biases to zero, so neurons learn the same features and this leads to poor training.

    Random initialization draws weights from normal or uniform distribution, however if range is too large, it can cause exploding or vanishing gradients.

    He initialization is commonly used for neural networks with ReLU-based activation function where weights are drawn from a normal distribution with a variance $\frac{2}{\text{number of input units}}$

4. What are some methods for dealing with the vanishing gradient problem?

    - Use appropriate weight initialization technique to ensure gradients start with reasonable magnitudes
    - Use non-saturating activation functions such as ReLU so that activations are not squashed into small ranges, ReLU is particularly effective due to its linear positive region
    - Use batch normalization as discussed earlier or layer normalization for sequential data
    - Use regularization as discussed earlier
    - Use LSTMs or transformers

5. What are some ways to deal with a model that overfits?

    - Apply regularization and dropout
    - Use augmentation to expand training dataset
    - Use early stopping by monitoring validation loss during training and stop when it stops improving

6. What metrics can be used for evaluating the model and give examples of when they can be used?

    Classification

    - Accuracy — proportion of correct predictions out of all predictions,
        - Predicting whether an email is spam or not with roughly equal numbers of spam and non-spam emails
    - Recall — proportion of true positives correctly identified out of all actual positives
        - Medical diagnosis where missing something is dangerous
    - Precision — Proportion of true positive predictions out of all positive predictions
        - Fraud detection, where falsely flagging a legitimate transaction is costly
    - F1 score — harmonic mean of precision and recall, helpful in imbalanced dataset
        - Sentiment analysis with imbalanced classes

    Regression

    - Mean Squared Error (MSE) — average of squared difference between predicted and actual values
        - House price preediction
    - Mean Absolute Error (MAE) — average of absolute difference between predicted and actual values

    Computer Vision

    - Intersection over Union (IoU) — ratio of intersection to the union of predicted and groud-truth label
        - Car detection
    - Mean Average Precision (mAP) — average precision across classes and IoU thresholds

## Convolutional Neural Networks

1. What is a convolution operation?

    A convolution operation is a fundamental technique used to process and analyze images. It involves sliding a kernel or filter, over an input image. At each position, the kernel performs element-wise multiplication with the overlapping region of the image, and the results are summed to produce a single output value. This process is repeated across the entire image to generate a feature map, which highlights specific patterns or features like edges, textures, or shapes. For example, a kernel might detect low-level features (e.g., edges) in early layers or high-level features (e.g., objects) in deeper layers.

2. What is a full, same and valid kernel?

    Full kernel refers to a convolution operation where the output is larger than the input image due to padding the input with zeros.

    Same kernel refers to a convolution operation where the output size is the same as the input size, achieved by adding appropriate padding to the input image.

    Valid kernel refers to a convolution operation where no padding is applied, and the kernel is only applied to positions where it fully overlaps with the input image.

    For $n \times n$ input and $k \times k$ kernel, the output size for

    - Full is $(n+k-1)(n+k-1)$
    - Same is $n \times n$
    - Valid is $(n-k+1)(n-k+1)$

3. What is the difference between convolution and correlation?

    Convolution and correlation are mathematical operations used in signal and image processing, often in the context of neural networks. While they are similar in that both involve sliding a kernel (or filter) over data to compute a weighted sum, they differ in how the kernel is applied. For convolution, kernel is flipped (rotated 180 degrees) before sliding over the input. For correlation, kernel is not flipped and applied as it is. In practice, frameworks skip the kernel flip since the kernel’s weights are learned during training, and flipping doesn’t affect the learned outcome.

4. How does backpropagation work in CNNs?

    Forward pass

    - Image → Conv → ReLU → Max Pool → Fully Connected → Softmax → Loss.

    Backward pass:

    - Compute loss gradient w.r.t. softmax output.
    - Backprop through fully connected layer to get gradients for weights and biases.
    - Backprop through max pooling by routing gradients to max positions.
    - Backprop through ReLU by applying the ReLU derivative.
    - Backprop through convolution by computing gradients for filters, biases, and input feature maps.
    - Update all parameters using the gradients.

5. What is transfer learning and when is it applied?

    Transfer learning is a machine learning technique where a model trained on one task is reused or fine-tuned for a different but related task. The pre-trained model provides learned features (e.g., edges, textures, or higher-level patterns) that can be leveraged, reducing training time and the need for large labeled datasets. Applied when labeled data is scarce, domains are similar, or faster training is needed, transfer learning reduces training time and improves performance by starting with robust features. It’s ideal for tasks like medical imaging or object detection with limited data but may be less effective if the source and target domains differ significantly or if a large target dataset is available for training from scratch.

6. Deepening: when is transfer learning useful, what alternatives do we have?

    Transfer learning is useful when training a model from scratch is impractical due to limited labeled data, computational resources, or time constraints. It leverages pre-trained models, typically trained on large, general datasets like ImageNet, to adapt to a specific task, such as classifying medical images or fine-tuning a language model for sentiment analysis. Alternatives to transfer learning include training a model from scratch, which is feasible with abundant data and resources but often impractical for smaller datasets due to overfitting risks. Another option is feature extraction, where pre-trained model layers are used as fixed feature extractors without fine-tuning, suitable for very small datasets but less flexible. Multi-task learning, where a model is trained on multiple related tasks simultaneously, can also be an alternative, though it requires careful task alignment and data availability. Lastly, data augmentation or synthetic data generation can mitigate data scarcity, but these methods may not capture the rich, generalizable features provided by transfer learning.

## Recurrent Neural Networks

1. What is a word embedding? Give some examples?

    Word embeddings are numerical vector representations of words that capture their meanings, relationships, and contexts in a continuous vector space. These dense representations aim to encode semantic and syntactic similarities between words, allowing machines to understand and process natural language more effectively. These representations are learned from large text corpora using techniques like Word2Vec, GloVe, and FastText. Word2Vec, for instance, predicts either the surrounding context words given a target word (Skip-Gram) or the target word based on its context (CBOW)

2. Explain what RNN is and what are some of its problems.

    RNN is a type of neural network designed to handle sequential data, making it well-suited for tasks like natural language processing, speech recognition, and time series analysis. Unlike feedforward networks, RNNs have loops that allow them to maintain a “memory” of previous inputs by using their hidden state, enabling them to process data with temporal dependencies. However, RNNs face several challenges, notably the vanishing gradient problem, where gradients diminish during backpropagation through time, making it difficult for the model to learn long-term dependencies. Conversely, the exploding gradient problem can lead to instability if gradients grow excessively. Additionally, RNNs can be computationally expensive and slow to train due to sequential processing.

3. What is an LSTM? How does it work?

    LSTM is a specialized type of RNN designed to address the vanishing and exploding gradient problems that hinder learning in traditional RNNs. LSTMs are capable of capturing long-term dependencies in sequential data by using a structure of gates to regulate the flow of information. Each LSTM cell contains three main gates: the forget gate, which decides which information to discard from the previous cell state; the input gate, which determines what new information to add; and the output gate, which decides what part of the cell’s state to output. These gates, combined with the cell state, allow LSTMs to selectively retain or forget information as needed, enabling them to model long-term dependencies effectively.

4. What is the difference between LSTMs and GRUs?

    LSTMs and GRUs are both advanced RNN architectures designed to address the limitations of traditional RNNs, particularly the vanishing gradient problem. While they share the goal of capturing long-term dependencies, they differ in their complexity and structure. LSTMs have three gates—forget, input, and output gates—along with a separate cell state to manage memory, offering fine-grained control over information flow. In contrast, GRUs simplify this process by using only two gates—reset and update gates—and combine the cell state and hidden state into a single representation, reducing the computational complexity. As a result, GRUs are typically faster to train and require fewer parameters, making them more efficient for certain tasks.

5. How does the attention mechanism work?

    The attention mechanism is a key concept in natural language processing and sequence-to-sequence models, designed to improve performance by allowing the model to focus on the most relevant parts of the input when generating an output. Instead of processing the entire input sequence uniformly, attention assigns varying weights to different parts of the sequence based on their relevance to the current task or output. This is achieved through a scoring function that computes alignment scores between the current decoder state and each encoder output. These scores are normalized using a softmax function to produce attention weights, which are then used to compute a weighted sum of the encoder outputs, referred to as the context vector. This context vector is used by the decoder to make predictions.

6. Can you explain how the transformer architecture works?

    The Transformer architecture replace recurrent layers with self-attention mechanisms, enabling parallelization and improved efficiency. At its core, the Transformer consists of an encoder-decoder structure, where the encoder processes the input sequence and generates a sequence of hidden representations, and the decoder uses these representations to generate the output sequence. Each encoder and decoder layer contains two main components: a multi-head self-attention mechanism, which allows the model to focus on different parts of the input sequence simultaneously, and a feed-forward neural network for further processing. Self-attention works by computing relationships between all tokens in a sequence, using key, query, and value vectors to capture dependencies regardless of their distance. Additionally, positional encodings are added to the input embeddings to preserve the sequence order.

7. What is BERT and what are some applications of BERT?

    BERT (Bidirectional Encoder Representations from Transformers) is a state-of-the-art natural language processing model developed by Google, designed to understand the context of words in text by analyzing their relationships both before and after the target word in a sentence. Unlike traditional models that process text in a unidirectional manner, BERT leverages a bidirectional transformer architecture to capture deep, nuanced contextual information. Its applications are vast and include tasks such as sentiment analysis, question answering (e.g., Google’s search engine improvements), named entity recognition, text classification, and summarization.

8. Explain how you could train a model for image captioning, VQA?

    Training a model for tasks like image captioning or visual question answering (VQA) involves combining computer vision and natural language processing techniques to bridge visual and textual understanding. Typically, a deep learning pipeline starts with a pre-trained convolutional neural network (CNN), such as ResNet or EfficientNet, to extract high-level visual features from the image. For image captioning, these features are input into a decoder, often based on RNNs, LSTMs, or Transformers, which generates textual descriptions by learning the relationship between visual features and corresponding sentences. For VQA, the visual features are combined with a representation of the input question, often encoded using pre-trained language models like BERT. A fusion mechanism, such as attention, integrates the visual and textual representations to focus on relevant parts of the image based on the question. The fused representation is then passed to a classification or prediction head to generate answers.

9. When would you want to use RNN, CNNs and Transformers? What are their strengths and weaknesses?

    RNNs are ideal for sequential data, such as time series, natural language, or speech, as they capture temporal dependencies through their recurrent structure. However, RNNs struggle with long-term dependencies due to the vanishing gradient problem and are computationally slow due to their sequential processing. CNNs excel at processing grid-like data, such as images, by leveraging convolutional layers to capture spatial hierarchies and local patterns. They are highly efficient and effective for tasks like image classification and object detection but are less suited for sequential or relational data. Transformers, on the other hand, are versatile and excel at capturing global dependencies in both sequential and non-sequential data using self-attention mechanisms. They are particularly effective for NLP tasks and are increasingly applied to vision tasks.

## General Adversarial Networks and Stable Diffusion

1. What is deconv?

    Deconvolution is the opposite of convolution, it upsamples or increases the spatial dimensions of a feature map. Typically implemented as a transposed convolution. Output size is determined by $\text{stride×(input size−1)+kernel size−2×padding}$

2. What is deep dream?

    Deep Dream is a fascinating technique developed by Google in 2015 that uses a convolutional neural network (CNN) to generate surreal,
    dream-like images by enhancing patterns the network "sees" in an input image

3. What is style transfer?

    Style transfer is a deep learning technique that combines the content of one image with the style of another image to create a new image. A content image and style image are taken as inputs. The images are then fed through the CNN and features from the context are being extracted from deeper layers which are rich in semantic information. The styles are extracted from multiple layers (often earlier ones) which captures textures, colors and patterns by measuring correlation between feature maps. Content loss measures how much the target's image contet deviates fromt he content image while style loss measures how much the target image's style deviates from the style image. The total loss is a weighted sum of the content loss and style loss. Gradient descent is used to iteratively modfiy the target image pixels to minimize the total loss.

4. What is GAN and how does it work?

    A deep learning framework that generates realistic data by having two neural netwroks, a generator and a discriminator go against each other. The generator creates fake data from random noise and attempts to fool the discriminator. THe discriminator judges whether data is real (from dataset) or fake (from generator). Over time, the generator improves its fakes to trick the discriminator while the discriminator gets better at spotting fakes. Both networks compete until the generator produces data that's nearly indistinguishable form the real data. For discriminator training, feed real data and label it as "real", feed fake data and label it as "fake", then update discriminator's weights to better distinguish real and fake by minimizing classification error. For generator training, pass the fake data to the discriminator and try to make the discriminator think its real (output close to 1), update generator's weights to fool the discriminator (minimize discriminator's ability to call it fake). The discriminator's loss is based on binary classification while the generator's loss is based on how well it tricks the discriminator. Train time iteratively and the generator should get better at producing realistic data and the discriminator gets better at spotting fakes. Ideally, they reach an equilibrium where the discriminator can’t tell real from fake (outputs ~0.5 for both).

5. What is stable diffusion and what separates it from normal diffusion?

    Diffusion models generate data by starting with random noise and gradually refining it into a clear output through step-by-step denoising process. Stable Diffusion is a specific type of diffusion model that runs the denoising process in a latent space instead of full iamge space, making it faster and less resource-intensive. Normal diffusion requires many denoising steps making generation slow and resource-intensive while stable diffusion uses fewer steps with optimized sampling. Normal diffusion is often unconditioned or conditioned to simple inputs while stable diffusion is designed for text conditioning. Normal diffusion uses a U-Net to denoise in pixel space while stable diffusion combines U-Net with pre-trained autoencoder and CLIP for latent diffusion and text guidance.

6. What are the differences between GAN and stable diffusion? What are the advantages and disadvantages of each of them?

    | **Aspect**                   | **GANs**                                                                                                                                                                  | **Stable Diffusion**                                                                                                                                        |
    | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | **Core Mechanism**           | Two networks: Generator creates fake data; Discriminator distinguishes real vs. fake. Adversarial minimax game.                                                           | Diffusion model: Iteratively denoises noise in latent space using a U-Net, guided by an autoencoder and CLIP.                                               |
    | **Training Process**         | Adversarial training; simultaneous Generator and Discriminator optimization. Prone to instability (e.g., mode collapse).                                                  | Non-adversarial; U-Net predicts noise in latent space. More stable but requires pretraining autoencoder.                                                    |
    | **Input & Conditioning**     | Random noise; conditional GANs use labels or limited text inputs. Less flexible for complex prompts.                                                                      | Random noise in latent space; excels at text conditioning via CLIP for complex prompts (e.g., “a dragon in a spacesuit”).                                   |
    | **Output Generation**        | Single forward pass through Generator; fast inference.                                                                                                                    | Iterative denoising (50–100 steps); slower but controllable.                                                                                                |
    | **Computational Efficiency** | Training is intensive; inference is fast (milliseconds). Requires large compute for stability.                                                                            | Training is heavy; inference is efficient in latent space, runs on consumer GPUs (~seconds).                                                                |
    | **Architecture**             | Generator uses transposed convolutions (deconv); Discriminator is typically a CNN.                                                                                        | U-Net for denoising; autoencoder (encoder + decoder) for latent space; CLIP for text.                                                                       |
    | **Applications**             | Face generation (StyleGAN), super-resolution (SRGAN), image-to-image (CycleGAN). Less flexible for text-to-image.                                                         | Text-to-image, inpainting, image-to-image, style transfer. Highly flexible for creative tasks.                                                              |
    | **Accessibility**            | Some open-source models (e.g., DCGAN); many are proprietary or research-focused.                                                                                          | Fully open-source (e.g., Hugging Face); pretrained weights widely available.                                                                                |
    | **Advantages**               | - Fast inference (~milliseconds).<br>- High-quality in specific domains (e.g., faces).<br>- Mature ecosystem since 2014.<br>- Compact Generator models.                   | - Stable training (no mode collapse).<br>- Efficient on consumer GPUs.<br>- Flexible text conditioning.<br>- Open-source and versatile.                     |
    | **Disadvantages**            | - Unstable training (mode collapse, vanishing gradients).<br>- Hard to train (needs tuning).<br>- Limited for complex text prompts.<br>- Ethical risks (e.g., deepfakes). | - Slower inference (~seconds).<br>- Complex pretraining.<br>- Text prompt dependency (poor prompts = poor results).<br>- Minor artifacts, ethical concerns. |

7. What is the difference between image captioning and VQA, explain both the concepts?

    Image captioning and VQA are 2 distinct tasks that involes understanding images and generating or processing text. While both combine visual and language understanding, they differ in objectives, inputs and outputs. Image captioning is the task of automatically generating a natural language description that summarizes the content of an image. VQA is the task of answering a natural language question about an image, requiring both visual understanding and reasoning.

    | **Aspect**             | **Image Captioning**                                                          | **Visual Question Answering (VQA)**                                                  |
    | ---------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
    | **Objective**          | Generate a descriptive caption summarizing the image’s content.               | Answer a specific question about the image, requiring targeted understanding.        |
    | **Input**              | Image only.                                                                   | Image + text question.                                                               |
    | **Output**             | A full sentence or phrase (e.g., “A dog runs in a field”).                    | A short answer, often a word, phrase, or sentence (e.g., “Blue”).                    |
    | **Task Type**          | Generative: Produces open-ended text.                                         | Answer prediction: Often classification or short text generation.                    |
    | **Processing**         | Vision model extracts features; language model generates a sequence of words. | Vision and language models process image and question; fusion model predicts answer. |
    | **Reasoning**          | Summarizes key elements without explicit reasoning.                           | Requires reasoning to interpret the question and extract relevant details.           |
    | **Output Scope**       | Broad, holistic description of the image.                                     | Narrow, specific response tied to the question.                                      |
    | **Training Data**      | Image-caption pairs (e.g., MS-COCO: image + “A cat on a mat”).                | Image-question-answer triplets (e.g., VQA v2: image + “What’s on the mat?” + “Cat”). |
    | **Evaluation Metrics** | BLEU, ROUGE, CIDEr (measure caption quality).                                 | Accuracy, F1 score (measure answer correctness).                                     |
    | **Complexity**         | Simpler, generates general descriptions.                                      | More complex, requires understanding question intent and image details.              |
    | **Example**            | Input: Image of a beach. Output: “Waves crash on a sandy beach.”              | Input: Image of a beach + “What color is the water?” Output: “Blue.”                 |

## Reinforcement Learning and TinyML

1. What is reinforcement learning and how is it different from supervised learning?
   Reinforcement Learning (RL) is a machine learning approach where an agent learns to make decisions by interacting with an environment to maximize a cumulative reward, using trial and error without explicit instructions. Unlike Supervised Learning, where a model is trained on labeled input-output pairs to minimize prediction errors, RL relies on sparse, delayed rewards rather than immediate labels. In RL, the agent explores actions, learns from feedback, and optimizes a policy for long-term goals. Supervised learning is suited for static prediction tasks like image classification. RL’s dynamic, sequential decision-making and lack of labeled data make it more complex but ideal for tasks requiring adaptability, while supervised learning’s reliance on labeled datasets makes it simpler and effective for well-defined prediction problems.

2. What is model based RL and model free RL?

    Model-based RL and model-free RL are 2 approaches within RL, differing in how the agent learns to make decisions in an environment to maximize cumulative rewards.

3. What is policy and value based RL?

    Policy-based and value-based reinforcement learning (RL) are two fundamental approaches to solving RL problems, each focusing on different aspects of decision-making. Policy-based RL directly learns a policy, which is a mapping from states to actions, specifying the probability of selecting each action in a given state. Value-based RL learns a value function, such as the Q-function, which estimates the expected cumulative reward for taking an action in a state and following the policy thereafter. The policy is derived indirectly by selecting actions that maximize this value, often using methods like Q-learning.

4. What is the difference between Q-learning and policy gradients?

    Q-learning, a value-based method, focuses on learning the Q-function, which estimates the expected cumulative reward for taking an action in a given state and following an optimal policy thereafter. It updates Q-values iteratively using the Bellman equation, adjusting them based on observed rewards and the maximum Q-value of the next state. Policy gradient methods, conversely, directly optimize a parameterized policy, such as a neural network, to maximize expected rewards by adjusting the probability distribution over actions.

5. Why would we use a discounted error ($\gamma$)?

    Used to compute the discounted cumulative reward, which weights future rewards less than immediate ones. This discounting ensures that the total reward remains finite, facilitating convergence of value functions in algorithms. By prioritizing near-term rewards, $\gamma$ reflects real-world scenarios where immediate outcomes are often more certain or valuable.

6. How does REINFORCE work?

    Policy gradient method in reinforcement learning that optimizes a parameterized policy, typically a neural network, to maximize the expected cumulative reward by directly adjusting the policy’s parameters. The process begins by sampling trajectories from the environment using the current policy, where each trajectory consists of states, actions, and rewards. For each time step, the algorithm computes the discounted return, which is the sum of future rewards weighted by the discount factor $\gamma$. Using these returns, REINFORCE calculates the policy gradient, which measures how the policy’s parameters should be adjusted to increase the likelihood of actions that yield higher returns. This gradient is derived from the expected reward, using the log-probability of the taken actions weighted by their corresponding returns. The policy parameters are then updated via gradient ascent, with a learning rate controlling the step size.

7. (Advanced) What is a replay buffer and what problem does it help us deal with?

    A replay buffer is a data structure used in off-policy reinforcement learning algorithms, such as Q-learning and Deep Q-Networks (DQN), to store and manage past experiences, represented as tuples of state, action, reward, and next state.

8. (Advanced) How is self-play overcoming the lack of exploration?

    This approach addresses the lack of exploration, a common challenge in RL where agents may get stuck in suboptimal strategies due to insufficiently diverse experiences. In self-play, the agent acts as both players in a game, generating a dynamic and evolving opponent as it improves. This creates a continuously challenging environment, as the opponent’s policy adapts to the agent’s progress, forcing the agent to explore new strategies to overcome stronger versions of itself.

9. Why do we quantize the models in TinyML and how does this affect the performance?

    Quantization converts high-precision weights and activations, typically 32-bit floating-point numbers, into lower-precision formats, such as 8-bit integers. This significantly reduces model size, enabling deployment on devices with limited storage, and lowers computational requirements, allowing faster inference with reduced power consumption. Additionally, integer operations are faster and more energy-efficient on embedded hardware compared to floating-point operations. However, quantization can impact performance by introducing approximation errors, as lower-precision representations lose some numerical accuracy. This may lead to a slight drop in model accuracy, particularly for tasks requiring fine-grained distinctions.

10. What are the two methods for quantization of a model and how are they different from each other?

    Post-training quantization (PTQ) and quantization-aware training (QAT). Post-training quantization takes a pre-trained model, typically with 32-bit floating-point weights, and converts its weights and activations to lower-precision formats, such as 8-bit integers, after training is complete. This process involves mapping the full-precision values to a quantized range, often using techniques like uniform quantization, and requires minimal additional training. Quantization-aware training incorporates the quantization process into the training pipeline, where the model is trained with simulated quantization effects, such as low-precision weights and activations, using techniques like fake quantization nodes. This allows the model to learn to compensate for quantization errors, resulting in better accuracy compared to PTQ, but it requires more computational resources and longer training times.

## Advanced topics

1. What is Artificial Curiosity and how does it work?

    Artificial curiosity is a concept in reinforcement learning (RL) and artificial intelligence where an agent is designed to explore its environment driven by intrinsic motivation, rather than solely relying on external rewards. It mimics human curiosity by encouraging the agent to seek novel or uncertain states, improving its understanding of the environment and enhancing exploration in sparse-reward settings. Artificial curiosity works by augmenting the reward function with an intrinsic reward, often based on measures like prediction error, novelty, or uncertainty. For example, an agent might receive a reward for visiting new states or reducing uncertainty in its world model, typically implemented using techniques like curiosity-driven exploration.

2. What is intrinsic motivation?

    intrinsic motivation is an agent’s drive to perform actions or explore an environment for internal reasons, rather than to achieve external rewards or goals. Unlike extrinsic motivation, where actions are driven by external rewards like scores in a game, intrinsic motivation encourages behaviors such as exploration, learning, or skill acquisition for their own sake.

3. How does Neuroevolution of Augmenting Topologies (NEAT) work?

    NEAT starts with a population of simple neural networks with minimal connections and iteratively evolves them by applying genetic operations: mutation, crossover, and selection. Mutation can adjust weights, add new connections, or insert new nodes, gradually increasing network complexity to match the task’s requirements. Crossover combines the structures of two parent networks while preserving compatibility, guided by historical markers that track the origin of network components to prevent destructive combinations. Selection favors networks with higher fitness, typically measured by task performance (e.g., cumulative reward in RL).

4. How does unsupervised task selection work?

    Unsupervised task selection is a technique in machine learning, particularly in reinforcement learning and unsupervised learning, where an agent autonomously selects tasks or goals to learn from without explicit human supervision. It aims to enable agents to explore and acquire skills in complex environments by generating their own objectives, often in the absence of predefined rewards.

5. What is hyperparameter optimization?

    Process of finding the best set of hyperparameters for a machine learning model to maximize its performance on a given task. Hyperparameters, such as learning rate, batch size, or the number of layers in a neural network, are set before training and significantly impact model accuracy, convergence, and efficiency. Typically involves defining a search space for hyperparameters and using optimization techniques to evaluate different configurations.

6. What is population-based hyperparameter optimization?

    Advanced hyperparameter optimization technique that evolves a population of models with different hyperparameters during training, inspired by evolutionary algorithms.

7. What is Neural Architecture Search (NAS)?

    Neural Architecture Search (NAS) is an automated process for designing neural network architectures optimized for a specific task, reducing the need for manual architecture engineering. NAS explores a search space of possible network configurations, including layer types, connections, and hyperparameters, to find architectures that maximize performance metrics like accuracy or efficiency.

8. What is AutoML?

    A set of techniques and tools that automate the end-to-end process of building machine learning models, making them accessible to non-experts and improving efficiency for experts. It is particularly valuable in industry settings, where rapid deployment of high-quality models is needed, and for democratizing machine learning by enabling users with limited expertise to build effective models.

9. What are alternatives if we only have few training data available?

    - Perform data augmentation to generate more data
    - Use pre-trained models as backbone and fine-tune on it

10. How does contrastive learning work?

    Contrastive learning is a self-supervised learning technique that trains models to distinguish between similar and dissimilar data points by comparing pairs of examples in a latent space. It aims to learn meaningful representations without labeled data by maximizing the similarity between positive pairs (e.g., augmented versions of the same image) while minimizing similarity between negative pairs (e.g., different images)

11. How does self supervised learning work?

    Self-supervised learning (SSL) is a machine learning paradigm that trains models to learn representations from unlabeled data by creating pretext tasks, where the supervision signal is derived from the data itself. Unlike supervised learning, which relies on labeled data, SSL generates pseudo-labels from the input data’s structure.

12. What is bias? Do we want it? What are some methods for dealing with bias?

    Bias in machine learning refers to systematic errors in a model’s predictions or decisions, often resulting from assumptions in the model or biases in the training data. Bias can manifest as unfair outcomes, such as a facial recognition system performing poorly on certain demographics due to underrepresented groups in the dataset. Some bias is intentional and desirable, such as inductive bias in model design (e.g., convolutional neural networks assume local spatial correlations in images), which helps models generalize. However, unwanted bias, such as societal biases reflected in data, is problematic, leading to unfair or inaccurate predictions. Methods to mitigate unwanted bias include data augmentation to balance dataset representation.

13. Where does unwanted bias come from?

    Unwanted bias in machine learning arises from multiple sources, primarily rooted in the data and modeling process. The most common source is biased training data, which reflects historical or societal inequities, such as underrepresentation of certain groups (e.g., gender or racial minorities) or skewed label distributions (e.g., biased hiring data).

14. What are the implications of GAN, stable diffusion and other generative AI methods? How can we mitigate those? Who is responsible?

    These models, capable of generating realistic images, text, or audio, enable applications like art creation, content generation, and data augmentation. However, they can amplify biases present in training data, producing outputs that reinforce stereotypes or exclude underrepresented groups. They also raise ethical concerns, such as generating deepfakes for misinformation, violating intellectual property by mimicking artistic styles, or enabling malicious uses like fraud. To mitigate these, techniques like fairness-aware training, differential privacy, and bias auditing can reduce biased outputs and protect data. Robust content moderation and watermarking generated outputs help combat misinformation, while legal frameworks can address intellectual property issues. Responsibility lies with multiple stakeholders: developers and researchers must design ethical models, companies deploying these systems should implement safeguards, policymakers should regulate misuse, and users must engage responsibly.

15. What is explainable AI (XAI) and what are the benefits of using it?

    XAI refers to techniques and methods that make the decisions or predictions of machine learning models interpretable and understandable to humans. XAI aims to provide insights into why a model makes specific predictions, how it processes inputs, or which features drive its outputs. For example, XAI can reveal why a loan approval model denied an application by highlighting influential factors like credit score. Benefits include improved trust, as users can verify model reasoning, especially in critical domains like healthcare or finance.

16. How does perturbation-based XAI work?

    It wokrs by explainig a model’s predictions by analyzing how changes (perturbations) to the input affect the output, revealing which input features are most influential. The process involves systematically altering parts of the input, such as occluding image patches, masking text tokens, or modifying numerical features, and observing the resulting changes in the model’s predictions.
