# Theory 3

## 3.1 Ethics

#### Background

Image captioning can be regarded as an end-to-end Sequence problem, as it
converts images, which are regarded as a sequence of pixels, to a sequence of
words. For this purpose, we need to process both the language or statements
and the images. We use recurrent neural networks for the language part, and
for the image part, we use convolutional neural networks to obtain the feature
vectors. Image Captioning
We are dealing with two types of information, a language one and another
image one. So, the question arises of how or in what order we should intro-
duce the information into our model. Elaborately speaking, we need a language
RNN model to generate a word sequence, so when should we introduce the im-
age data vectors in the language model? A paper by Marc Tanti and Albert
Gatt [Comparison of Architectures], Institute of Linguistics and Language Tech-
nology, University of Malta covered a comparison study of all the approaches.
Image Captioning
You can read the post and Andrej Karpathy’s Architecture.
Then, you should be able to complete the following tasks.

### 3.1.1 Pros and Cons of Utilising Concatenation for Combining Embeddings
Concatenation is a straightforward and widely used technique. It involves appending the two sets of embeddings into a single, longer vector, which is then fed into the next layer of the model
|Pros|Cons|
|-|-|
|Simplicity — concatenation is computationally and conceptually simple and no need for complex transformations <br/> <br/> Flexibility — works with embedding of different sizes and dosen't require them to be aligned in dimensionality | Increased dimensionality — can lead to higher computational costs as there are more parameters and lead to the curse of dimensionality where the data becomes sparse in high dimensional space, making it harder to generalize <br/><br/>  Lack of Semantic Integration — dosen't inherently model the relationship between image and language embeddings and might require more layers of capacity to learn the dependencies

Concatenation is a lightweight, no-frills option that shines in its simplicity and preservation of raw information, making it a good baseline for combining embeddings in image captioning or similar tasks. However, its high dimensionality and lack of inherent fusion mean it often needs to be paired with a robust downstream model to truly shine.

### 3.1.2 Pros and Cons of Utilising Addition for Combining Embeddings
Addition combines them into a single vector of the same dimensionality by summing their elements. This requires the embeddings to have the same size.

|Pros|Cons|
|-|-|
|Integration — blends information for both modalities into a single representation, allow model to learn a unified feature space <br/><br/>  Dimensionality Preservation — addition dosen't increase the size of resulting vector and reduces computational complexity|Information Loss — can obscure individual contributions from each modality and cancel out overlapping signals and potentially losing nuance <br/><br/> Sensitive to Scaling — addition assumes embeddings are on the same scale, if image features have larger magnitude than word embeddigs (vice versa), one modality can dominate the result unless careful normalization is applied |

Addition offers an efficient, compact way to combine embeddings, ideal for scenarios where a unified representation is desired without inflating dimensionality. However, its simplicity comes at the cost of potential information loss and sensitivity to embedding compatibility.

### 3.1.3 Pros and Cons of Utilising Multiplication for Combining Embeddings
Corresponding elements of two vectors are multiplied together to produce a new vector of the same dimensionality. Like addition, this requires the embeddings to have the same size.

|Pros|Cons|
|-|-|
|Non-Linear Fusion — multiplication introduces non-linearity into combination process and can help model capture more complex relationship between modalities <br/><br/> Selective Emphasis — multiplication acts like a dynamic mask if one modality has a near zero value in a dimension, effectively nullifying the other modality in that dimension| Information Loss — on one hand, it can be beneficial, on the other it can be destructive if embeddings need to be preserved <br/><br/> Sensitivity — output is highly sensitive to magnitude of inputs, if one embedding has large values and other has small ones, result can explore or shrink dramatically leading to numerical instability or vanishing gradients|

Multiplication offers a compact, non-linear way to combine embeddings, excelling at modeling interactions and selective emphasis between modalities. Its gating-like behavior can be a strength in tasks needing tight coupling, but it comes with risks of information loss, scale sensitivity, and reduced flexibility.

### 3.1.4 Pros and Cons of Utilising Attention for Combining Embeddings
Attention doesn’t statically merge the embeddings into a single vector. Instead, it dynamically weighs and integrates features from one modality based on their relevance to the other.

|Pros|Cons|
|-|-|
|Dyanmic Relevance — attention allows model to focus on relevant parts of one modality depending on the context of the other, making it more expressive than static methods <br/><br/> Flexibility — attention doesn't require the embeddings to have the same dimensionality| Computational Complexity — expensive and can slow training and inference time especially with large inputs <br/><br/> Interpretability — not straightforward as high weights don't always correlate with human intuition and the "black-box" makes it debugging tricky too|

Attention is a powerful, flexible approach for combining embeddings, excelling at capturing context-specific relationships and boosting performance in complex multimodal tasks like image captioning. However, its computational cost, training demands, and complexity make it a heavyweight solution compared to the simplicity of concatenation, addition, or multiplication.

### 3.1.5 Pros and Cons of Utilising Difference for Combining Embeddings
Take two vectors and compute their element-wise difference, resulting in a new vector of the same dimensionality.

|Pros|Cons|
|-|-|
|Contrastive Learning — good for scenarios where dissimilarity needs to be learned|Infornation Loss — important shared information can be cancelled out if embeddings have similar values in certain dimensions|

Subtraction is a lightweight, difference-focused method for combining embeddings, offering simplicity and efficiency while emphasizing discrepancies between modalities. However, its destructive nature, lack of cooperative fusion, and limited applicability make it a poor fit for most multimodal tasks like image captioning, where synthesis is key.

