# 📜 Document Denoising for Archival Papers

**Denoising Autoencoder (U-Net) · PyTorch · Kaggle Dataset**

Removes coffee stains, ink bleeds, and noise from scanned historical documents using a Denoising Convolutional Autoencoder trained on the Kaggle **Denoising Dirty Documents** dataset.

---

## Demo

| Noisy (Input)    | Denoised (Output) |
| ---------------- | ----------------- |
| noisy image here | clean image here  |

---

## Architecture

```text
Input (Noisy Document)
        │
        ▼
┌─────────────────────────────────────────┐
│        U-Net Denoising Autoencoder      │
│                                         │
│ Encoder:    [32] → [64] → [128] → [256]│
│ Bottleneck: [512]                       │
│ Decoder:    [256] → [128] → [64] → [32]│
│ Skip connections at each level          │
│                                         │
│ Total Parameters: ~7.7M                 │
└─────────────────────────────────────────┘
        │
        ▼
Output (Clean Document)
```

**Loss:** `0.5 × MSE + 0.5 × (1 - SSIM)`
**Metrics:** `PSNR (dB) ↑` and `SSIM ↑`

---

## Project Structure

```text
document-denoising/
│
├── Document_Denoising_Colab.ipynb      # Train on Google Colab
├── src/
│   ├── model.py                        # U-Net architecture
│   ├── dataset.py                      # Patch-based dataset loader
│   ├── train.py                        # Training loop + checkpointing
│   └── utils.py                        # PSNR, SSIM, inference utils
│
├── frontend/
│   ├── app.py                          # Flask web app
│   └── templates/
│       └── index.html                  # Upload UI
│
├── outputs/
│   └── best_model.pth                  # Place downloaded model here
│
├── checkpoints/                        # Auto-created during training
├── requirements_colab.txt
├── requirements_frontend.txt
└── README.md
```

---

## Quick Start

### Train on Google Colab

1. Open `Document_Denoising_Colab.ipynb` in Google Colab
2. Set runtime to **GPU (T4)**
3. Run cells top to bottom
4. Upload `kaggle.json` when prompted
5. Training runs for **100 epochs (~90–120 min)**
6. `best_model.pth` downloads automatically at end

**Auto-resume supported:**
If Colab disconnects, rerun notebook — training resumes from latest checkpoint.

---

### Run Frontend Locally

```bash
git clone https://github.com/Adhi-raaj/document-denoising.git
cd document-denoising

pip install -r requirements_frontend.txt
python frontend/app.py
```

Open browser at:
`http://localhost:5000`

---

## Training Details

| Setting         | Value                        |
| --------------- | ---------------------------- |
| Model           | U-Net Denoising Autoencoder  |
| Parameters      | ~7.7M                        |
| Patch Size      | 128 × 128                    |
| Patches / Image | 40                           |
| Batch Size      | 32                           |
| Epochs          | 100                          |
| Optimizer       | Adam (lr=1e-3, cosine decay) |
| Loss            | 0.5 × MSE + 0.5 × (1 − SSIM) |
| Target PSNR     | > 30 dB                      |
| Training Time   | 90–120 min (Colab T4)        |

---

## Checkpoint Resume

Saved every epoch:

* `checkpoints/latest.pth`
* `checkpoints/epoch_XXXX.pth`

Contains:

* Model weights
* Optimizer state
* Scheduler state
* Training history
* Timestamp

---

## Local Inference Performance (CPU)

| Image Size  | Inference Time |
| ----------- | -------------- |
| 420 × 540   | ~1–2 sec       |
| 800 × 1000  | ~4–6 sec       |
| 1200 × 1600 | ~10–15 sec     |

---

## Kaggle Dataset Setup

1. Go to Kaggle → Account
2. Create New API Token
3. Download `kaggle.json`
4. Upload it in notebook Step 3

Competition: **Denoising Dirty Documents**

---

## Tech Stack

* **Model:** PyTorch, U-Net
* **Training:** Google Colab, Google Drive
* **Metrics:** PSNR, SSIM
* **Frontend:** Flask, Vanilla JavaScript
* **Dataset:** Kaggle – Denoising Dirty Documents

