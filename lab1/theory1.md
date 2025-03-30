# Theory 1

## 1.1 Ethics

### Background

Bias in training data is a pertinent issue that demands the attention of
machine learning practitioners and data scientists. It pertains to the presence
of skewed, unrepresentative, or unfair elements within the data used to train
machine learning models. When present, bias can lead to inaccurate or unfair
predictions and decisions made by the model. To gain a deeper understanding
of bias in training data, consider the following points:

Skewed Representation: Overrepresentation or underrepresentation of cer-
tain groups or types of data in the training dataset can result in biased out-
comes. For example, if a facial recognition system is trained mostly on images
of lighter-skinned individuals, it may not perform well in accurately identifying
people with darker skin tones.

Unfair Treatment: Bias in training data can lead to unfair treatment of
certain groups. This is particularly concerning in areas such as credit scoring,
hiring processes, and law enforcement.

Impact on Model Performance: Biased training data can significantly impact
the performance of machine learning models, leading to inaccurate predictions
and decisions.

Ethical Considerations: Addressing bias in training data is an integral part
of ethical AI development, as it aims to ensure that AI systems make fair and
unbiased decisions for all individuals and groups.

You can ask any LLM of your choice and they will tell you why they are
biased! It all comes down to the training data fed. Here we are trying to show
you how they display their bias.

First take any LLM you want and feed them some professions prompts: like
the doctor, nurse, the lawyer, the office worker, the janitor, the construction
worker, etc. translate them into non-neutral (like english) into another with
genders (Swedish, Spanish, etc.) and try to get it to use stereotypes.
Present your findings with examples.

### Answer

| Profession          | Korean                          | Description (Korean)                                                                                    | Spanish                   | Description (Spanish)                                                                            |
| ------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------ |
| Doctor              | 의사 (uisa)                     | "The doctor is a tall, smart man with glasses. He always wears a white coat and saves patients.”        | El doctor                 | “A male doctor, authoritative, probably wearing a white coat and barking orders in the ER.”      |
| Nurse               | 간호사 (ganhosa)                | "The nurse is a pretty, kind woman. She helps the doctor and smiles warmly at patients."                | La enfermera              | “A female nurse, nurturing, sweet, and perpetually overworked while doting on patients.”         |
| Lawyer              | 변호사 (byeonhosa)              | "The lawyer is a sharp-eyed man in a suit. He wins arguments in the courtroom."                         | El abogado                | “A slick, aggressive male attorney in a suit, chasing ambulances or arguing in court.”           |
| Office Worker       | 사무직 직원 (samujik jigwon)    | "The office worker is a boring man who sits at a computer drinking coffee. He only looks at paperwork." | La oficinista             | “A female desk worker, gossiping at the water cooler, obsessed with paperwork and coffee.”       |
| Janitor             | 청소부 (cheongsobu)             | "The janitor is an older woman. She quietly sweeps the floor with a broom."                             | El conserje               | “A gruff male custodian, mopping floors and grumbling about the mess everyone leaves behind.”    |
| Construction Worker | 건설 노동자 (geonseol nodongja) | "The construction worker is a sweaty, muscular man. He wears a helmet and smashes walls with a hammer." | El obrero de construcción | “A burly male laborer, whistling at passersby and eating a sandwich on a steel beam.”            |
| Product Manager     | 제품 관리자 (jepum gwanrija)    | "The product manager is a young man in trendy clothes. He pitches ideas in meetings."                   | El gerente de producto    | “A male tech-bro type, obsessed with Agile sprints and throwing around buzzwords like "synergy." |

The descriptions of office workers and janitors reveal clear gender and behavioural stereotypes tied to these professions. The office worker is portrayed as a disengaged man focused on coffee and paperwork in one context, while in another, a female office worker is depicted as gossiping and socially engaged. These portrayals trivialise office work and reinforce gendered assumptions about workplace behaviour. Similarly, the janitor is described as an older, quiet woman sweeping the floor, or as a male custodian mopping and complaining about others’ messes. These characterisations perpetuate biases associating women with domestic, submissive roles and men with physical labor and irritability. These biases can lead to unfair treatment, reinforcing societal stereotypes and limiting opportunities for underrepresented groups in these professions. Moreover, such biases can impact model performance, causing AI systems to make inaccurate or stereotyped predictions about individuals based on their profession.

## 1.2 Metrics

### Background

As a fundamental tool for evaluating classification models, the confusion
matrix, also known as an error matrix, provides a square table that displays
the number of correct and incorrect predictions made by a model for each tar-
get class. The matrix is structured with rows representing the actual class and
columns representing the predicted class.

The confusion matrix represents four key metrics: True Positives (TP), True
Negatives (TN), False Positives (FP), and False Negatives (FN). These metrics
are useful in measuring the model’s performance and can be used to calculate
performance measures like accuracy, precision, recall, and F1-score.

The confusion matrix offers several advantages in evaluating classification
models. Firstly, it provides a clear and interpretable visualization of the model’s
performance, allowing for quick identification of strengths and weaknesses in
classifying specific classes. Additionally, it facilitates error analysis, enabling
researchers to pinpoint areas for improvement and refine the model’s training
process. The matrix is also particularly beneficial when dealing with imbalanced
class distributions, where one class might have significantly more instances than
others.

In conclusion, the confusion matrix is a valuable tool for evaluating classifica-
tion models because it provides a clear and detailed representation of the model’s
performance. Its metrics enable researchers to identify areas for improvement
and refine the model’s training process, making it an essential element of the
machine learning toolkit.

Fill out all the missing values, and put an explanation of why accuracy may not be the best metric.

### Answer

Accuracy is defined as the ratio of correctly predicted instances to total instances, measures overall correctness of predictions assuming dataset is balanced

Precision is defined as the ratio of correctly predicted positive instances to the total predicted positive instances, focuses on quality of positive predictions

Recall is defined as the ratio of correctly predicted positives instances to the actual positive instances, used to measure ability to identify all relevant instances

F1 score is the harmonic mean of precision and recall, balancing the two metrics, useful when class distribution is imbalanced, providing a single metric to balance false positives and false negatives

Accuracy may not be the best metric to use especially if the dataset is unbalanced. For example, a disease occurs in only 1% of cases and a model predicts “no disease” for all cases achieving 99% accuracy but it is completely ineffective at diagnosing the disease.

$$
\begin{array}{c|cc}
\text{} & \text{Predicted Negative} & \text{Predicted Positive} \\
\hline
\text{Actual Negative} & 990 & 10 \\
\text{Actual Positive}  & 20  & 30 \\
\end{array}
$$

| Metric                                                                 | Value                                          |
| ---------------------------------------------------------------------- | ---------------------------------------------- |
| True Negatives (TN)                                                    | 990                                            |
| False Positives (FP)                                                   | 10                                             |
| False Negatives (FN)                                                   | 20                                             |
| True Positives (TP)                                                    | 30                                             |
| Accuracy $$\frac{TP + TN}{TP+TN+FP+FN}$$                               | $$\frac{30+990}{30+990+10+20}=0.9714$$         |
| Recall $$\frac{TP}{TP+FN}$$                                            | $$\frac{30}{30+20}=0.6$$                       |
| Precision $$\frac{TP}{TP+FP}$$                                         | $$\frac{30}{30+10}=0.75$$                      |
| F1 Score $$\frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$$ | $$\frac{2\cdot0.75\cdot0.6}{0.75+0.6}=0.6666$$ |

$$
\begin{array}{c|cc}
\text{} & \text{Predicted Negative} & \text{Predicted Positive} \\
\hline
\text{Actual Negative} & 9000 & 50 \\
\text{Actual Positive}  & 100  & 850 \\
\end{array}
$$


| Metric                                                                 | Value                                                    |
| ---------------------------------------------------------------------- | -------------------------------------------------------- |
| True Negatives (TN)                                                    | 9000                                                     |
| False Positives (FP)                                                   | 50                                                       |
| False Negatives (FN)                                                   | 100                                                      |
| True Positives (TP)                                                    | 850                                                      |
| Accuracy $$\frac{TP + TN}{TP+TN+FP+FN}$$                               | $$\frac{850+9000}{850+9000+50+100}=0.985$$               |
| Recall $$\frac{TP}{TP+FN}$$                                            | $$\frac{850}{850+100}=0.8947$$                             |
| Precision $$\frac{TP}{TP+FP}$$                                         | $$\frac{850}{850+50}=0.9444$$                            |
| F1 Score $$\frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$$ | $$\frac{2\cdot0.9444\cdot0.8947}{0.9444+0.8947}=0.9188$$ |
