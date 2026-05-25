# Project: Edge-Optimized VLM for Degraded Text (DeepSeek Visual Primitives)
**Timeline:** June 1 - August 1 (8 Weeks)
**Objective:** Fine-tune SmolVLM-500M using Knowledge Distillation to perform DeepSeek-style spatial Chain-of-Thought for drone OCR.

## Phase 1: Infrastructure & Data Formatting (Weeks 1-2)
- [ ] **Environment Setup:** Update CachyOS, install latest CUDA drivers, PyTorch, and Hugging Face `TRL` (Transformers Reinforcement Learning) library.
- [ ] **Data Ingestion:** Load the 4,013 images from the IJCNN degraded text dataset.
- [ ] **Coordinate Normalization:** Convert existing SA-DBNet bounding box coordinates into the discrete integer format `[0-999]` required for the `<box>` tokens.
- [ ] **Prompt Engineering:** Format the training data into the specific JSON/Message structure: `[User Prompt] -> [Thinking: <ref>text</ref><box>[[x1,y1,x2,y2]]</box>] -> [Final OCR Answer]`.

## Phase 2: Teacher Generation & Distillation Prep (Week 3)
- [ ] **Teacher Setup:** Load a large frontier model via API (or run Qwen2-VL-72B/LLaVA if compute allows).
- [ ] **Trace Generation:** Feed the formatted dataset into the Teacher model to generate perfect, step-by-step spatial reasoning traces for all 4,000+ images.
- [ ] **Data Filtering:** Scrub the Teacher outputs. Discard any traces where the bounding box coordinates are hallucinated or the OCR text is wrong. 

## Phase 3: Fine-Tuning the Student (Weeks 4-5)
- [ ] **Baseline Training:** Run a direct supervised fine-tuning (SFT) run on SmolVLM-500M without the Teacher's traces (just Image -> Output). Save checkpoint.
- [ ] **Knowledge Distillation:** Fine-tune SmolVLM-500M using the pristine Teacher traces. Freeze the vision encoder (as seen in the Small Docling paper) to maintain stability.
- [ ] **Hyperparameter Tuning:** Run small batches testing learning rates (e.g., 2e-4 vs 5e-5) and LoRA ranks (r=16 vs r=64).

## Phase 4: Benchmarking & Optimization (Weeks 6-7)
- [ ] **Metrics Calculation:** Test models on the unseen validation split. Calculate Exact Match Accuracy, Word Error Rate (WER), and Character Error Rate (CER).
- [ ] **A/B Testing:** Compare the Direct Fine-Tuned model vs. the Distilled model. 
- [ ] **Edge Profiling:** Quantize the winning model to 8-bit. Measure VRAM footprint and inference FPS to prove viability for low-compute drone hardware.

## Phase 5: Paper Writing & Formatting (Week 8)
- [ ] **Draft Methodology:** Detail the distillation process and why DeepSeek visual primitives solve the degraded VQA bottleneck.
- [ ] **Draft Results:** Create comparative tables highlighting the throughput (FPS) vs. Accuracy trade-off against your old SA-DBNet pipeline.
- [ ] **Final Review:** Wrap up the draft for mentor review before the August semester begins.