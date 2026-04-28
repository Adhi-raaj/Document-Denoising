📜 Document Denoising for Archival Papers
Denoising Autoencoder (U-Net) · PyTorch · Kaggle Dataset
Removes coffee stains, ink bleeds, and noise from scanned historical documents using a Denoising Convolutional Autoencoder trained on the Kaggle Denoising Dirty Documents dataset.

Demo
Noisy (Input)	Denoised (Output)
noisy	clean
Architecture
Input (Noisy Document)
    │
    ▼
┌─────────────────────────────────────────┐
│   U-Net Denoising Autoencoder           │
│                                         │
│  Encoder:  [32] → [64] → [128] → [256] │
│  Bottleneck: [512]                      │
│  Decoder:  [256] → [128] → [64] → [32] │
│  Skip connections at each level        │
│                                         │
│  Total Parameters: ~7.7M               │
└─────────────────────────────────────────┘
    │
    ▼
Output (Clean Document)
Loss: 0.5 × MSE + 0.5 × (1 - SSIM)
Metrics: PSNR (dB) ↑ and SSIM ↑

Project Structure
document-denoising/
│
├── Document_Denoising_Colab.ipynb   ← Train on Google Colab
│
├── src/
│   ├── model.py                     ← U-Net architecture
│   ├── dataset.py                   ← Patch-based dataset loader
│   ├── train.py                     ← Training loop + checkpointing
│   └── utils.py                     ← PSNR, SSIM, inference utils
│
├── frontend/
│   ├── app.py                       ← Flask web app (run locally)
│   └── templates/
│       └── index.html               ← Upload UI
│
├── outputs/
│   └── best_model.pth               ← Place downloaded model here
│
├── checkpoints/                     ← Auto-created during training
│
├── requirements_colab.txt           ← Colab training deps
├── requirements_frontend.txt        ← Local frontend deps
└── README.md
Quick Start
1. Train on Google Colab
Open Document_Denoising_Colab.ipynb in Google Colab
Set runtime to GPU (T4): Runtime → Change runtime type → T4 GPU
Run cells top to bottom:
Mounts Google Drive (checkpoints auto-saved there)
Downloads Kaggle dataset (you need kaggle.json API key)
Trains for 100 epochs (~90–120 min)
Downloads best_model.pth at the end
If Colab disconnects mid-training: Just re-run from Step 1. The notebook auto-resumes from the last saved epoch via Google Drive.

2. Run Frontend Locally
# Clone the repo
git clone https://github.com/YOUR_USERNAME/document-denoising.git
cd document-denoising

# Install frontend dependencies
pip install -r requirements_frontend.txt

# Place your trained model
# Copy best_model.pth → outputs/best_model.pth

# Start the Flask app
python frontend/app.py
Open your browser at http://localhost:5000

Upload any scanned document image → get the denoised version back in seconds.

Training Details
Setting	Value
Model	U-Net Denoising Autoencoder
Parameters	~7.7M
Patch size	128 × 128
Patches / image	40 (random, augmented)
Batch size	32
Epochs	100
Optimizer	Adam (lr=1e-3, cosine decay)
Loss	0.5 × MSE + 0.5 × (1 − SSIM)
Target PSNR	> 30 dB
Estimated time	90–120 min (Colab T4 GPU)
Checkpoint Resume
Every epoch saves:

checkpoints/latest.pth — always the newest
checkpoints/epoch_XXXX.pth — one per epoch
Both locally (Colab) and on Google Drive (persistent)
Each checkpoint contains: model weights, optimizer state, scheduler state, full training history, and a timestamp.

To resume, just re-run the training cell with CONFIG['resume'] = True (default).

Local Inference Performance (CPU)
Image Size	Inference Time
420 × 540	~1–2 seconds
800 × 1000	~4–6 seconds
1200 × 1600	~10–15 seconds
Uses overlapping-patch inference with Hanning window blending to avoid boundary artifacts.

Kaggle Dataset Setup
Go to kaggle.com → Account → Create New API Token
Download kaggle.json
Upload it when prompted in the Colab notebook (Step 3)
The competition is: denoising-dirty-documents

Tech Stack
Model: PyTorch, U-Net
Training: Google Colab (GPU), Google Drive
Metrics: PSNR, SSIM
Frontend: Flask, vanilla JS
Dataset: Kaggle — Denoising Dirty Documents
