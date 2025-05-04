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
