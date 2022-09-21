# qubit-routing
Making efficient qubit routing using a modified DDQN that evaluates states.

This project is a continuation of the work done in the bachelor's project [TIFX04-22-20](https://github.com/karieinarsson/TIFX04-Kompilering-av-kvantdatorkod).

## Requirements

To be able to run the training you need to install these python3 packages:
 - gym
 - torch
 - pygame
 - tensorboard
 - pandas

Install them using the requirements.txt file:

`pip3 install -r requirements.txt`

## Usage

### Training a model we use "Train.py"

Train.py is where the model is trained and all variables for the training are defined

`Train.py`

### Visulizing training results

`python3 -m tensorboard.main --logdir=logdir`

### Visualizing quantum circuit sulotion

`Visualize.py`

### Batch training

 BatchTrain.py

 python3 BatchTrain.py -j <jsonfile> -s <jsonscheme file>
