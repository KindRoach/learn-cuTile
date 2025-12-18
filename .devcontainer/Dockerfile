FROM nvidia/cuda:13.1.0-devel-ubuntu24.04

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip python-is-python3
RUN rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED

# Install Pytorch
RUN pip install torch --index-url https://download.pytorch.org/whl/cu130

# Install cuTile python
RUN pip install cuda-tile

# Install git
RUN apt-get install -y git
