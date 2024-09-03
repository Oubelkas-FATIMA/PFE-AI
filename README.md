# Satellite Imagery Pipeline

## Introduction 
This repository houses the Satellite Imagery PFE, encompassing various functionalities, including:

- Satellite Search Engine (Download Satellite Images)
- Crop Classification Pipeline 
- Field boundary detection


## Getting Started
- Install Miniconda

  **Linux**:
  ```
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  chmod +x Miniconda3-latest-Linux-x86_64.sh
  ./Miniconda3-latest-Linux-x86_64.sh
  ```
- Source your bash to make conda available
  ```
  source ~/.bashrc
  ```

- Create an environment for the project
  ```
  conda create -n satellite-pfe python=3.8
  ```

- Install the Python requirements and pre-commit hooks. Press y when asked. This may take a few minutes
  ```
  conda activate satellite-pfe
  make install
  ```

  If running on a linux machine, you may need to also install libgl1
  ```
  apt update; apt install -y libgl1
  ```