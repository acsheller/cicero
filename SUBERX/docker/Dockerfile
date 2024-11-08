# Use an official NVIDIA CUDA image with Python 3.11
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up environment variables to pass user ID and group ID
# Create a non-root user named 'suber' with specific UID and GID
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} devuser && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash devuser && \
    echo "devuser:devuser" | chpasswd && \
    adduser devuser sudo


# Switch to the new user
USER devuser

# Set the working directory to the home directory
WORKDIR /home/devuser

# Create a Python virtual environment in the home directory
RUN python3.11 -m venv /home/devuser/venv

# Ensure that the virtual environment is activated on container startup
ENTRYPOINT ["/bin/bash", "-c", "source /home/devuser/venv/bin/activate && exec bash"]

# By default, run an interactive shell
CMD ["bash"]
