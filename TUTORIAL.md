# DefectFill2 Tutorial

This document provides a comprehensive guide to the **DefectFill2** codebase, an unofficial implementation of the paper *"DefectFill: Realistic Defect Generation with Inpainting Diffusion Model for Visual Inspection"*.

This tutorial explains the core concepts, the model architecture, the training process, and how to use the code for defect generation.

---

## 1. Introduction & Concept

**DefectFill** is designed to generate realistic industrial defects on "good" (non-defective) object images. This is crucial for training visual inspection systems where defective samples are often scarce.

The core idea relies on **Stable Diffusion Inpainting**, but with specific modifications to ensure:
1.  **Defect Realism**: The generated defect looks like a real defect (e.g., a crack, a stain).
2.  **Object Integrity**: The non-defective parts of the object remain unchanged and consistent.
3.  **Precise Control**: The defect appears exactly where requested (via a mask).

### Key Mechanisms
To achieve this, the method uses three loss components during training:
1.  **Defect Loss ($L_{def}$)**: Learns to reconstruct real defects given a mask.
2.  **Object Loss ($L_{obj}$)**: Learns to preserve the object's background and structure using random masks.
3.  **Attention Loss ($L_{attn}$)**: Forces the model to "attend" to the defect concept (e.g., "broken") specifically within the masked region.

---

## 2. Project Structure

- `model.py`: Defines the model architecture, including the custom attention processor.
- `train.py`: The main training loop implementing the dual-path training strategy.
- `inference.py`: Script for generating defects on new images.
- `data_loader.py`: Handles MVTec AD dataset loading and processing.
- `utils.py`: Checkpointing utilities.

---

## 3. Code Breakdown

### 3.1. The Model (`model.py`)

The `DefectFillModel` wraps a pre-trained `Stable Diffusion Inpainting` pipeline. To adapt it for defect generation without destroying the pre-trained knowledge, it uses **LoRA (Low-Rank Adaptation)**.

#### Initialization
The model loads Stable Diffusion, freezes the VAE (Variational Autoencoder), and injects LoRA layers into the UNet and Text Encoder.

```python
# model.py
class DefectFillModel(nn.Module):
    def __init__(self, ...):
        # ... load Stable Diffusion ...

        # Configure LoRA for UNet
        unet_lora_config = LoraConfig(
            r=lora_rank,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            ...
        )

        # Apply LoRA
        self.pipeline.unet = get_peft_model(self.pipeline.unet, unet_lora_config)

        # Register custom attention processor
        self.register_attention_processor()
```

#### Attention Control (`AttentionStoreProcessor`)
A key innovation is the **Attention Loss**. To compute this, we need access to the internal cross-attention maps of the UNet. The `AttentionStoreProcessor` intercepts these maps during the forward pass.

```python
# model.py
class AttentionStoreProcessor(AttnProcessor):
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, ...):
        # ... standard attention calculation ...

        # Store cross-attention maps
        if is_cross_attention and self.model is not None:
             # Look for "attn2" (cross-attention) in "up_blocks" (decoder)
            if "attn2" in name and "up_blocks" in name:
                self.model.attention_maps[name] = reshaped_probs.detach().clone()

        # ... return hidden states ...
```

#### Attention Loss
The `get_attention_loss` function ensures that the attention for the defect keyword (e.g., "broken") is focused inside the defect mask.

```python
# model.py
def get_attention_loss(self, masks):
    # ...
    # Get attention map for the defect token (e.g., "broken")
    defect_attn = attn_map[b, :, :, defect_token_idx].mean(dim=0)

    # Resize to match mask size
    resized_attn = F.interpolate(...)

    # Minimize MSE between Attention Map and Defect Mask
    # This forces the model to look at the masked area when processing the "defect" word.
    sample_loss = F.mse_loss(avg_attn_map, mask)
    return attention_loss
```

---

### 3.2. Training Strategy (`train.py`)

The training loop uses a **Dual-Path Strategy**. For every batch of defect images, it performs two passes with different objectives.

#### Path 1: Defect Learning
Uses the **real defect mask** and a prompt describing the defect (e.g., "A photo of broken_large"). This teaches the model what the defect looks like.

```python
# train.py (Simplified)

# 1. Real Mask Path
real_mask_prompts = [f"A photo of {defect_type}" ...]

# Forward pass with real masks
outputs = model(..., masks=defect_masks)

# Calculate Defect Loss (Reconstruction)
defect_loss = model.get_defect_loss(...)

# Calculate Attention Loss (Alignment)
attention_loss = outputs.get("attention_loss")
```

#### Path 2: Object Integrity Learning
Uses a **random rectangular mask** (which likely covers healthy parts of the object) and a prompt describing the object with the defect (e.g., "A bottle with broken_large"). Since the image is the same (contains the real defect elsewhere), masking a random healthy part and asking the model to reconstruct it teaches the model to preserve the background and not hallucinate defects everywhere.

```python
# train.py (Simplified)

# 2. Random Mask Path
random_mask_prompts = [f"A {obj_class} with {defect_type}" ...]

# Forward pass with random masks
obj_outputs = model(..., masks=random_masks)

# Calculate Object Loss
object_loss = model.get_object_loss(...)
```

#### Total Loss
The final loss is a weighted sum:

```python
total_loss = lambda_defect * defect_loss + lambda_obj * object_loss + lambda_attn * attention_loss
```

Typical weights: `lambda_defect=0.5`, `lambda_obj=0.2`, `lambda_attn=0.05`.

---

### 3.3. Inference (`inference.py`)

During inference, we want to generate a defect on a clean image.

1.  **Input**: Clean image, Mask (where to create defect), Defect Type (e.g., "scratch").
2.  **Process**:
    - The model generates multiple samples (e.g., 8) using the prompt "A {object} with {defect}".
    - It uses **LPIPS (Learned Perceptual Image Patch Similarity)** to select the best sample.
3.  **Selection Metric**:
    The code selects the sample with the **highest LPIPS score** compared to the original clean content *within the masked region*.

    *Why highest?*
    We are comparing `Original (Clean) * Mask` vs `Generated (Defect) * Mask`. A high LPIPS score means the generated content is significantly different from the original clean surface, implying a visible defect has been successfully generated.

```python
# inference.py
lpips_score = model.lpips_model(
    clean_tensor_model_format * mask_resized,
    sample * mask_resized
).mean()

if lpips_score > best_score:
    best_score = lpips_score
    best_sample = sample
```

---

## 4. Dataset Preparation (`data_loader.py`)

The code expects the **MVTec AD** dataset structure:

```
root/
  <object_class>/ (e.g., bottle)
    train/
      defective/
        <defect_type>/ (e.g., broken_large)
          000.png
      defective_masks/
        <defect_type>/
          000_mask.png
```

The `MVTecDefectDataset` class handles loading these pairs. For the training set, it loads real defects. For the test set (or if masks are missing), it can generate random masks.

---

## 5. How to Run

### 1. Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Training
To train a model on the "bottle" class:
```bash
python train.py \
  --data_dir ./MVTec \
  --object_class bottle \
  --output_dir ./models/bottle \
  --batch_size 2
```

### 3. Inference
To generate a "broken_large" defect on a specific image:
```bash
python inference.py \
  --checkpoint ./models/bottle/checkpoint_final.pt \
  --object_class bottle \
  --image_path ./MVTec/bottle/test/good/001.png \
  --mask_path ./my_mask.png \
  --defect_type broken_large
```

---

## 6. Summary

This implementation provides a robust framework for defect generation by:
1.  Leveraging the strong prior of **Stable Diffusion**.
2.  Using **LoRA** for efficient fine-tuning.
3.  Employing **Attention Control** to bind defect semantics to specific regions.
4.  Using **Dual-Path Training** to balance defect generation with background preservation.
