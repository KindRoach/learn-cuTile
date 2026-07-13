FROM nvcr.io/nvidia/pytorch:26.06-py3

# config user and bash completion
RUN apt-get update && apt-get install -y sudo bash-completion \
    && echo "ubuntu ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu

RUN pip install --upgrade cuda-tile[tileiras]

# The WSL dev container mounts its driver libraries here.
ENV LD_LIBRARY_PATH="/usr/local/wsl/lib:${LD_LIBRARY_PATH}"
