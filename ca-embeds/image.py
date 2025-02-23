import os
from pathlib import Path
import modal
from dotenv import load_dotenv

# load the .env file
load_dotenv()

# Hugging Face Token
HF_TOKEN = os.getenv("HF_TOKEN")

# CUDA image
CUDA_VERSION = "12.4.0"
CUDA_FLAVOR = "devel"  # devel includes full CUDA toolkit
UBUNTU_VERSION = "ubuntu22.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{UBUNTU_VERSION}"

# environment variables
image_env = {
    "BUILD_WITH_CUDA": "true",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "CUDA_VISIBLE_DEVICES": "0",
    "HF_TOKEN": HF_TOKEN,
}

# linux packages
linux_packages = [
        "build-essential",
        "cmake",
        "g++",
        "clang",
        "git",
        "git-lfs",
        "ninja-build",
        "pkg-config"
]

# Build comprehensive image with proper CUDA support
image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{CUDA_TAG}",
        add_python="3.12"
    )
    # Environment setup first - needed for subsequent steps
    .env(image_env)
    # System dependencies using apt_install
    .apt_install(
        *linux_packages
    )
    # Build dependencies
    .pip_install([
        "ninja",
        "packaging",
        "wheel",
        "setuptools"
    ])
    # Core ML dependencies with CUDA support
    .pip_install([
        "torch",
        "torchvision",
        "torchaudio",
    ], extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install([
        "accelerate",
        "transformers"
    ])
    # General python libraries
    .pip_install([
        "python-dotenv",
        "pydantic",
        "fastcore",
        "sentence-transformers",
        "huggingface_hub[hf_transfer]",
        "fastapi[standard]",
        # for putting the embeddings elsewhere
        "supabase",
        "vecs",
    ])
    # Install flash-attn with CUDA support
    .run_commands(
        "CXX=g++ MAX_JOBS=8 pip install flash-attn --no-build-isolation --no-cache-dir"
    )
    # Install latest transformers
    .run_commands(
        "pip install -U git+https://github.com/huggingface/transformers.git"
    )
)