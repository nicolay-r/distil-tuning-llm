#!/bin/bash
cd
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

# Create an empty file named setup-get.sh in the current directory
touch setup-git.sh

# Create a directory named app_keys and an empty file named k.txt inside it
mkdir -p app_keys
touch app_keys/k.txt

# create conda environments for fine-tuning
conda env create -f environment.yml




# cd evaluates
# conda env create -f environment.yml
