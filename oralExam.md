## General Questions

1. What regularization techniques do you know?

    L1 regularization adds absolute values of the model's weights to the loss function by driving some weights to zero.
    L2 regularization adds squared values of model's weights to the loss function by penalizing large weights, encourage smaller and more evenly distributed weights.
    Dropout can be used to randomly deactivate neurons during training

2. What is batch normalization?

    Batch normalization is a technique used in neural networks to stablilize and accelerate training by normalizing the inputs to each layer. It works by adjusting and scaling the activations with a mini-batch to have a mean of zero and a variance of one. Using batch normalization stabilizes training and allows the the network to converge faster.

3. What initialization methods do you know?

    Zero initialization sets all weights and biases to zero, so neurons learn the same features and this leads to poor training. Random initialization draws weights from normal or uniform distribution, however if range is too large, it can cause exploding or vanishing gradients. He initialization is commonly used for neural networks with ReLU-based activation function where weights are drawn from a normal distribution with a variance $\frac{2}{\text{number of input units}}$

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

Prepare for questions where you are given a model and the result and you have to determine strengths and weaknesses, we might also ask you to come up with an alternative solution

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

## Advanced topics

Artificial Curiosity
What is Artificial Curiosity and how does it work?
What is intrinsic motivation?
AutoML and Meta-Learning
How does Neuroevolution of Augmenting T opologies (NEAT) work?
How does unsupervised task selection work?
What is hyperparameter optimization?
What is population-based hyperparameter optimization?
What is Neural Architecture Search (NAS)?
What is AutoML?
Learning with few data
What are alternatives if we only have few training data available?
How does contrastive learning work?
How does self supervised learning work?
Ethics and Bias
What is bias? Do we want it? What are some methods for dealing with bias?
Where does unwanted bias come from?
What are the implications of GAN, stable diffusion and other generative AI
methods? How can we mitigate those? Who is responsible?
Explainable AI
What is explainable AI (XAI) and what are the benefits of using it?
How does perturbation-based XAI work?

## Open-Ended Questions

What will advance the area of DL in the future? Hardware, architectures,
training methods, something else?
How is it possible for a human to evaluate a machine when the human
accuracy is less than 100%?
How much data does the human brain need to be smart?
What could we improve in the course?
What could you have improved?
