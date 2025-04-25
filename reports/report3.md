# Report 3

Describe what you have learned during this study period. You can also include references to external material you have used to enhance your learning process (e.g., book chapters, scientific papers, online courses, online sites).

| Task     | Time Spent (hours) |
| -------- | ------------------ |
| Lectures | 6                  |
| Labs     | 12                 |

## Stable Diffusion

This week, I explored the concepts of Variational Autoencoders (VAEs) and U-Net architectures, focusing on their applications in image processing, alongside an understanding of Stable Diffusion. A VAE, incorporating a U-Net with a cross-attention mechanism, transforms images from pixel space to a latent representation through its encoder, downsampling to reduce complexity. The U-Net within the VAE denoises the latent representation iteratively, reversing diffusion steps to reconstruct a sharp image via the decoder. U-Net, named for its U-shaped structure, is designed for image segmentation. Its encoder extracts high-level semantic features by reducing spatial dimensions through repeated convolutional blocks, max pooling, and doubling the number of channels after each downsampling. At the bottleneck, the feature map has minimal spatial dimensions but rich semantic content due to numerous channels. The decoder restores spatial resolution using upsampling, convolutional layers to refine features, and halving the number of channels per upsampling, producing a pixel-wise output while preserving semantic information. Skip connections between the encoder and decoder enhance feature integration. Stable Diffusion, a related technique, leverages a similar diffusion-based approach, using a U-Net to iteratively denoise a latent representation starting from random noise, guided by a text prompt through cross-attention mechanisms to generate high-quality images. Unlike traditional VAEs, Stable Diffusion operates in latent space for efficiency and is trained on large datasets to produce diverse, photorealistic outputs, making it a powerful tool for text-to-image generation.

I have also dedicated a fair amount of time to work on the group project on car detection in snow.
