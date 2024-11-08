# Use an appropriate base image
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    rsync \
    curl \
    git \
    git-lfs \
    wget \
    sudo \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda && \
    rm /tmp/miniconda.sh

# Set PATH to include Miniconda
ENV PATH="/opt/miniconda/bin:$PATH"

# Create a non-root user named 'suber' with specific UID and GID
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} suber && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash suber && \
    echo "suber:suber" | chpasswd && \
    adduser suber sudo

# Create the SUBER directory and set permissions
RUN mkdir -p /home/suber/SUBER && chown -R suber:suber /home/suber/SUBER

# Switch to the new user
USER suber
WORKDIR /home/suber/SUBER

# Copy the requirements.txt file into the container
COPY requirements.txt pythonrc.py /tmp/

# Initialize conda for bash
RUN /opt/miniconda/bin/conda init bash && /opt/miniconda/bin/conda create -n MPR python=3.9.0 -y && \
    /opt/miniconda/bin/conda run -n MPR conda install -c nvidia/label/cuda-11.8.0 cuda -y && \
    /opt/miniconda/bin/conda run -n MPR conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.8 -y

# Export necessary environment variables and install Python requirements
RUN echo "source /opt/miniconda/etc/profile.d/conda.sh" > set_env_vars.sh && \
    echo "conda activate MPR" >> set_env_vars.sh && \
    echo "export TORCH_CUDA_ARCH_LIST='6.0;6.1;7.0;7.2;7.5;8.0;8.6+PTX;8.9;9.0'" >> set_env_vars.sh && \
    echo "export PYTHONSTARTUP='/tmp/pythonrc.py'" >> set_env_vars.sh && \
    echo "export GITHUB_ACTIONS=true" >> set_env_vars.sh && \
    echo "pip install -r /tmp/requirements.txt" >> set_env_vars.sh && \
    chmod +x set_env_vars.sh && \
    /bin/bash set_env_vars.sh

RUN echo '#!/bin/bash ' > /home/suber/entrypoint.sh && \
    echo 'source /opt/miniconda/etc/profile.d/conda.sh' >> /home/suber/entrypoint.sh && \
    echo 'conda activate MPR' >> /home/suber/entrypoint.sh && \
    echo 'exec "$@"' >> /home/suber/entrypoint.sh && \
    chmod +x /home/suber/entrypoint.sh

EXPOSE 5003
EXPOSE 8888
# The code to run when container is started:
ENTRYPOINT ["/home/suber/entrypoint.sh"]
# Default command
CMD ["bash"]
