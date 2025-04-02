# Vehicle Detection on Snow

Vehicle detection is a critical component of computer vision applications such as autonomous driving, traffic surveillance, and road safety systems. While significant progress has been made in detecting vehicles under normal weather conditions, adverse environments like snow pose unique challenges that degrade the performance of standard detection models. Snowy conditions introduce factors such as reduced visibility, occlusion of vehicle features by snow accumulation, low contrast between vehicles and their surroundings, and variable lighting due to reflections or overcast skies. These issues can obscure key visual cues, making it difficult for traditional algorithms to accurately locate and classify vehicles.

The project focuses on developing a deep learning-based solution to detect vehicles in snowy environments using the Nordic Vehicle Dataset (NVD). Proposed by Homam Mokayed at LTU, this project leverages the NVD, a specialized dataset designed to address the challenges of vehicle detection under adverse weather conditions, particularly snow. The dataset, accessible at https://nvd.ltu-ai.dev/, includes detailed annotations and a variety of snowy scenarios, making it an ideal resource for training robust detection models.

## Problem

This project is an object detection task where the goal is to locate and classify objects within images. The snowy environment introduces challenges such as low visibility, occlusion by snow, and varying lighting conditions, making it a complex real-world problem. The dataset’s focus on Nordic conditions ensures the model must generalize across diverse snow-related scenarios.

## Suitable Architectures

### YOLO11

Single-stage object detector renowned for its balance of speed and accuracy. For this project, we plan to use YOLO11 as the primary architecture, leveraging its pre-trained weights and fine-tuning on the NVD to adapt to snow-specific challenges.

### DETR

Transformer-based architecture instead of traditional convolutional networks. Unlike YOLO, DETR is an end-to-end model that eliminates the need for anchor boxes and NMS, instead treating detection as a set prediction problem. DETR’s strength lies in its global reasoning capability, which could be advantageous for detecting cars in complex snowy scenes with occlusions or overlapping objects. For this project, DETR could serve as an experimental alternative to explore if transformer-based attention improves performance in low-visibility conditions.

### Faster R-CNN

Faster R-CNN, a two-stage object detector from the R-CNN family, remains a benchmark for high-accuracy detection tasks. It consists of a Region Proposal Network (RPN) to generate candidate regions and a second stage to refine these proposals and classify objects. Its two-stage design allows it to focus computational resources on promising regions, potentially improving performance in scenes with heavy snow occlusion.

## Loss Functions

| **Architecture** | **Bounding Box Regression Loss** | **Classification Loss** |
| ---------------- | -------------------------------- | ----------------------- |
| **YOLO11**       | Mean Squared Error (MSE)         | Cross-Entropy           |
| **DETR**         | L1 Loss                          | Cross-Entropy           |
| **Faster R-CNN** | Smooth L1 Loss                   | Cross-Entropy           |

## Evaluation Metrics

To assess the model’s performance, standard object detection metrics will be used.

Mean Average Precision (mAP): Calculated at an IoU threshold (e.g., 0.5), mAP is the primary metric to evaluate precision and recall across the test set. It reflects the model’s ability to detect all cars accurately.

Precision and Recall: These will provide insights into false positives (e.g., mistaking snow piles for cars) and false negatives (e.g., missing obscured cars).

F1 Score: A harmonic mean of precision and recall to summarize overall performance.

Inference Speed: Measured in frames per second (FPS), this ensures the model is practical for real-time use, a key consideration for YOLO-based systems.

## Accelerator Estimations

| **Architecture** | **Estimated GPU VRAM** | **Notes**                                                      |
| ---------------- | ---------------------- | -------------------------------------------------------------- |
| **YOLO11**       | 8–12 GB                | Efficient single-stage design; lower memory needs              |
| **DETR**         | 16–24 GB               | Transformer-based; high memory due to attention mechanisms     |
| **Faster R-CNN** | 12–16 GB               | Two-stage model; moderate memory with region proposal overhead |

## Possible Challenges

### Data Preprocessing

Snow introduces noise and variability in image quality, requiring robust preprocessing techniques like normalization or augmentation. Inconsistent image resolutions or annotation quality in the NVD could further complicate data preparation.

### Feature Extraction Difficulty

Low-contrast scenes, where vehicles blend with snowy backgrounds challenge the model’s ability to extract distinctive features. Convolutional layers or transformer attention mechanisms may struggle to identify edges or shapes, necessitating deeper or specialized backbones.
