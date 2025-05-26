#!/bin/bash

# Update package list
apt-get update

# Install required packages
apt-get install -y libxml2 build-essential

# Navigate to the parent directory
cd ..

# Download and run the CUDA installer
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run
sh cuda_12.0.0_525.60.13_linux.run 

# Navigate to the distill-d2n directory
cd root/distill-d2n

# create conda environments for fine-tuning
conda env create -f environment.yml

# create conda environments for evaluation
cd evaluate
conda env create -f environment.yml
