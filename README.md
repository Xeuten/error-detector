# About

This project represents a detector of errors in audio files. It is intended to detect errors of different types and distinguish them from each other.

# Prerequisites

## Software

- Python 3.12
- ffmpeg
- CUDA toolkit 12.1
- libomp140.x86_64.dll (for Windows)

## Hardware

- Nvidia GPU with CUDA support with at least 10GB of VRAM (or 6 GB if you switch to a smaller model)
- At least 8 gb of RAM

# Installation

```
cd /path/to/project/
python -m venv venv
pip install -r requirements.txt
```
- Fill the .env file according to the .env.example file
