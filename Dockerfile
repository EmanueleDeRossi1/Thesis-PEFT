# 🔹 Use Ubuntu as base image (compatible with ARM64 & x86)
FROM ubuntu:22.04

# 🔹 Set working directory inside the container
WORKDIR /app

# 🔹 Install system dependencies
RUN apt-get update && apt-get install -y python3 python3-pip git

# 🔹 Copy dependency files first (to leverage Docker caching)
COPY requirements.txt requirements-cuda.txt install.sh ./

# 🔹 Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 🔹 Detect system and install CUDA dependencies if available
RUN chmod +x install.sh && ./install.sh || echo "Skipping CUDA installation."

# 🔹 Copy the rest of the project files into the container
COPY . .

# 🔹 Default command: Run training
CMD ["python3", "scr/training/trainer.py"]