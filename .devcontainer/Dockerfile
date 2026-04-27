FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies for Go engine, C++ builds, and tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnugo \
    cmake \
    build-essential \
    python3-dev \
    git \
    curl \
    rclone \
    libeigen3-dev \
    zlib1g-dev \
    sudo \
    ca-certificates \
    jq \
    fzf \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 (Claude Code installed later as dev user)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Docker CLI (for building/pushing images via host Docker socket)
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu jammy stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y --no-install-recommends docker-ce-cli docker-buildx-plugin \
    && rm -rf /var/lib/apt/lists/*

# Create non-root dev user (UID=1000 for bind mount compatibility)
RUN useradd -m -s /bin/bash -u 1000 dev \
    && echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && cp -r /root/.local /home/dev/.local \
    && chown -R dev:dev /home/dev/.local

RUN groupadd -f docker && usermod -aG docker dev

USER dev
ENV PATH="/home/dev/.local/bin:/home/dev/.npm-global/bin:$PATH"
RUN npm config set prefix /home/dev/.npm-global \
    && npm install -g @anthropic-ai/claude-code
# Set `DEVCONTAINER` environment variable to help with orientation
ENV DEVCONTAINER=true

# Install Python 3.10 as dev user so paths resolve correctly
RUN uv python install 3.10

# Place venv outside /workspace so bind-mount doesn't clobber it
ENV UV_PROJECT_ENVIRONMENT=/home/dev/.venv
WORKDIR /workspace

# Copy build dependencies (actual install happens in postCreateCommand)
COPY --chown=dev:dev pyproject.toml uv.lock ./
COPY --chown=dev:dev scripts/ scripts/
COPY --chown=dev:dev src/alpha_go/cpp/ src/alpha_go/cpp/

# Additional tools (appended to preserve layer cache)
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    tmux \
    git-lfs \
    rsync \
    && sudo rm -rf /var/lib/apt/lists/*
