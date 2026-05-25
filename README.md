# Edge-VLM for Degraded Drone OCR

**Author:** Aditya Sharma  

## Overview
This repository contains the training pipeline and experimental framework for fine-tuning sub-1B parameter Vision-Language Models (SmolVLM) to perform text-centric Visual Question Answering (VQA) on highly degraded images. 

Traditional Multimodal Large Language Models (MLLMs) experience logical collapse when reading motion-blurred or distorted text because natural language Chain-of-Thought (CoT) lacks strict spatial grounding. This project solves that "Reference Gap" by implementing DeepSeek's **"Thinking with Visual Primitives"** methodology on extreme edge-device hardware.

## Methodology

### 1. Spatial Visual Primitives
Instead of predicting OCR text directly, the model is trained to generate spatial tokens (`<ref>` and `<box>`) internally before answering. By outputting discrete integer coordinates `[0-999]` that physically bound the text in the image, the model anchors its cognitive trajectory to the exact pixel locations of the degraded text.

### 2. Synthetic Drone Degradation Pipeline
To simulate real-world drone telemetry, human-verified scene text datasets (e.g., ICDAR, COCO-Text) are programmatically degraded using the `Albumentations` library. We inject:
* **Motion Blur:** Simulating high-speed pitch/yaw shifts.
* **Defocus/Blur:** Replicating variable depth-of-field lens lag.
* **Gaussian Noise:** Simulating low-light sensor artifacts.

### 3. Curriculum Learning Framework
To prevent gradient instability in the 500M parameter model, training is divided into two sequential stages:
* **Stage 1 (The Locator):** Supervised Fine-Tuning (SFT) specifically for mapping visual pixels to normalized bounding box coordinates.
* **Stage 2 (The Reasoner):** Utilizing the adapted weights from Stage 1, the model is trained on complex VQA pairs to perform spatial Chain-of-Thought reasoning.

## Hardware Optimization (6GB VRAM Strategy)
This pipeline is engineered to run on consumer hardware (RTX 4050 6GB) for prototyping, and scaled to dual-T4 cloud instances for full dataset execution.
* **Quantization:** 4-bit NormalFloat (NF4) base weights.
* **Adaptation:** QLoRA targeted strictly at attention projection modules (`q_proj`, `v_proj`).
* **Memory Management:** Gradient Checkpointing and Paged AdamW 8-bit optimizers to prevent CUDA Out-of-Memory (OOM) errors via system RAM paging.