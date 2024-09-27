# Ataro Project - Reinforcement Learning by Moshe Yelisevitch and Tal Grossman

### Important note:
The code was trained and tested on Ubuntu. It is recommended to run the code on this operating system, since we fixed the environment for this OS. If you are using Windows, it might not work properly.
## Required installations

- create and activate your virtual environment and run the following commands (conda for example):
```
conda create -n <your_env_name> python=3.11.9
conda activate <your_env_name>
```
- install the following packages:
```
pip uninstall gym
pip install "gym[atari]"==0.9.5
pip install opencv-python
```
also make sure you have torch installed. If not, install using PyTorch installation instructions [here](https://pytorch.org/get-started/locally/)

## How to run the code
- To run the code, you need to run the following command in your terminal:
1. `cd` to the directory of the project
2. run ``` python env_fix_pong.py``` , in order to fix the environment of the atari games.
3. run ``` python main.py ``` 




