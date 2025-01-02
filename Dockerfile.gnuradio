# Start with NVIDIA's development container
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# Preconfigure timezone and install dependencies
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

RUN apt-get update && apt-get install -y \
    software-properties-common cmake g++ python3 python3-pip python3-venv git sudo \
    python3-packaging python3-setuptools python-is-python3 build-essential usbutils libusb-1.0-0-dev \
    xfce4 xfce4-terminal x11vnc xvfb novnc python3-websockify supervisor wget curl \
    filezilla vim udev python3-mako terminator ncdu gr-osmosdr librtlsdr-dev rtl-sdr \
    gnuradio gnuradio-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Google Chrome
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && apt-get install -y google-chrome-stable && \
    rm -rf /var/lib/apt/lists/*

# Install Visual Studio Code
RUN wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /usr/share/keyrings/packages.microsoft.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/vscode stable main" > /etc/apt/sources.list.d/vscode.list && \
    apt-get update && apt-get install -y code && \
    rm -rf /var/lib/apt/lists/*

# Install noVNC and WebSockify
RUN mkdir -p /opt/novnc/utils/websockify && \
    wget -qO- https://github.com/novnc/noVNC/archive/v1.5.0.tar.gz | tar xz --strip-components=1 -C /opt/novnc && \
    wget -qO- https://github.com/novnc/websockify/archive/v0.12.0.tar.gz | tar xz --strip-components=1 -C /opt/novnc/utils/websockify

# Create non-root user 'guser' and set permissions
RUN useradd -ms /bin/bash guser && \
    getent group dialout || groupadd -g 20 dialout && \
    usermod -aG dialout guser && \
    mkdir -p /home/guser/.vnc /etc/supervisor/conf.d && \
    chown -R guser:guser /home/guser /etc/supervisor/conf.d /opt/novnc && \
    echo "guser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Add udev rule for the RTL-SDR device
RUN echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2838", MODE="0666", GROUP="dialout"' > /etc/udev/rules.d/20-rtlsdr.rules

# Set working directory
WORKDIR /app
RUN chown -R guser:guser /app

# Configure Supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Set environment variables
ENV PATH="/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV PYTHONOPTIMIZE=1
ENV TORCH_CUDA_ARCH_LIST='6.0;6.1;7.0;7.2;7.5;8.0;8.6+PTX;8.9;9.0'
ENV TF_CPP_MIN_LOG_LEVEL=3

# Expose necessary ports
EXPOSE 8080 5900

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh

# Make the script executable
RUN chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]


# Start Supervisor
CMD ["/usr/bin/supervisord"]
