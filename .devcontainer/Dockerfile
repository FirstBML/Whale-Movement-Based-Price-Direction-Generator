# Devcontainer base (includes Python + common tools)
FROM mcr.microsoft.com/devcontainers/python:1-3.11

# Install Docker CLI so Codespaces can run Docker commands
USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends docker.io && \
    rm -rf /var/lib/apt/lists/*

USER vscode
