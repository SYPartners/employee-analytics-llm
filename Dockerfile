
# Use a stable NVIDIA CUDA base image for PyTorch and deep learning
# This image includes CUDA, cuDNN, and the necessary drivers for DGX Spark
FROM nvcr.io/nvidia/pytorch:24.05-py3

# Set environment variables
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# We use pip install --no-cache-dir to keep the image size down
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the entrypoint for the container. The actual command will be passed via `docker run`
# or the `launch_distributed_training.sh` script.
ENTRYPOINT ["/bin/bash", "-c"]

