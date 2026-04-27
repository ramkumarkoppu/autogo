#!/bin/bash
set -e

# Ensure reliable DNS — systemd-resolved's stub (127.0.0.53) doesn't work
# reliably inside Docker containers. Add real upstream DNS servers.
if grep -q '127.0.0.53' /etc/resolv.conf 2>/dev/null; then
    echo "Configuring systemd-resolved with upstream DNS..."
    sudo mkdir -p /etc/systemd/resolved.conf.d
    printf "[Resolve]\nDNS=8.8.8.8 8.8.4.4\n" | sudo tee /etc/systemd/resolved.conf.d/dns.conf >/dev/null
    sudo systemctl restart systemd-resolved
fi

# Install Docker engine if not present or broken
if ! command -v docker &>/dev/null || ! docker info &>/dev/null 2>&1; then
    echo "Installing Docker (removing any broken pre-existing install)..."
    sudo systemctl stop docker docker.socket containerd 2>/dev/null || true
    sudo apt-get remove -y docker docker-engine docker.io docker-ce docker-ce-cli containerd runc 2>/dev/null || true
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo tee /etc/apt/keyrings/docker.asc >/dev/null
    sudo chmod a+r /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
        sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
    # Start with clean config to avoid inheriting broken daemon.json
    echo '{}' | sudo tee /etc/docker/daemon.json >/dev/null
    sudo systemctl enable docker
    sudo systemctl start docker
else
    echo "Docker already installed and running"
fi

# Ensure the calling user can talk to the docker socket without sudo.
# New group membership won't apply to the current SSH session, so callers
# that run immediately after setup should use `sudo docker` or `sg docker`.
if [ "$(id -u)" != "0" ] && ! id -nG | grep -qw docker; then
    sudo usermod -aG docker "$(whoami)"
fi


# Install NVIDIA Container Toolkit on GPU machines
if command -v nvidia-smi &>/dev/null; then
    if ! dpkg -l nvidia-container-toolkit &>/dev/null; then
        echo "Installing NVIDIA Container Toolkit..."
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
    else
        echo "NVIDIA Container Toolkit already installed"
    fi
fi
