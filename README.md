# Playing Flappy Bird with RL

**Tom LABIAUSSE** - March 2024

[Click here to access the notebook of the project](demo.ipynb)

This project implements and compare Monte Carlo and Sarsa($\lambda$) agents on the game of Flappy Bird with the specific environment defined [here](https://gitlab-research.centralesupelec.fr/stergios.christodoulidis/text-flappy-bird-gym).

## Setup

Install the requirements:
```bash
pip install -r requirements.txt
```

Install the flappy bird environments:
```bash
pip install git+https://gitlab-research.centralesupelec.fr/stergios.christodoulidis/text-flappy-bird-gym.git
```

## Files

* `configs.py` : global project configuration
* `agents.py`: implementation of tabular RL agents
* `trainers.py`: implementation of trainers for the agents
* `utils.py`: utility functions for visualization
* `main.py`: launch training/agent demo

## Training & Demo

You can either use the provided [notebook](demo.ipynb) which contains all the code of this project or use the following command lines.

* Train a Monte Carlo agent:
```bash
python main.py --mc
```

* Train a Sarsa($\lambda$) agent:
```bash
python main.py --sarsa-lambda
```

* Demo using a saved agent:
```bash
python main.py --mc --demo
python main.py --sarsa-lambda --demo
```

Change all environment/training parameters directly in `main.py`.

## Results preview

### MC Agent

<img src='./figures/mc_svalues_iso.png' width='500'>
<img src='./figures/mc_svalues_top.png' width='500'>

### Sarsa($\lambda$) Agent

<img src='./figures/sarsa_svalues_iso.png' width='500'>

<img src='./figures/sarsa_svalues_top.png' width='500'>