# Question 2 
## Required installations
- create and activate your virtual environment and run the following commands (conda for example):
```
conda create -n <your_env_name> python=3.11.9
conda activate <your_env_name>
```
- In order to run 'policy_iter.py' you need to install gym version 0.10.5. You can do this by running the following commands in your terminal:

```
pip uninstall gym
pip install gym==0.10.5
```
- also make sure you have the following packages installed: numpy, matplotlib
## How to run the code
- To run the code, you need to run the following command in your terminal:
``` python policy_iter.py ``` 
or
``` python3 policy_iter.py ```
## Output files
- This folder contains 2 output files:
    - 'IterationTable.png': contains a screenshot of V[0] vs iteration table.
    - 'VvsIter.png': contains a figure of value vs iteration for each state.
    


