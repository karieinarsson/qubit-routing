# qubit-routing
Making efficient qubit routing using a modified DDQN that evaluates states.

This project is a continuation of the work done in the bachelor's project [TIFX04-22-20](https://github.com/karieinarsson/TIFX04-Kompilering-av-kvantdatorkod).

## Usage

### Training a model we use "Train.py"

Train.py is where the model is trained and all variables for the training are defined

`Train.py`

### Visulizing training results

`tensorboard --logdir logdir/`

### Visualizing quantum circuit sulotion

`Visualize.py`

### Batch training

 BatchTrain.py

 python3 BatchTrain.py -j <jsonfile> -s <jsonscheme file>
