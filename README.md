# NEAT-Pong

## Setup
Download the code using the following command:
```
$ git clone  https://github.com/domonik/neat_Pong
```


The dependencies can be installed using:
 
```
$ pip install -r "requirements.txt"
```


## Basic Idea
### NEAT
The idea behind the program is to get first experience with the 
[python integration](https://neat-python.readthedocs.io/en/latest/) of ["Evolving Neural Networks through
Augmenting Topologies"](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf). The program can be used to train an
AI opponent for a version of the classical Pong Game.

The Basic idea behind NEAT is mirroring evolution by randomly creating or deleting nodes as well as randomly 
deleting/creating connections between such nodes in an Artificial Neural Network (NN). This mirrors deletions and 
insertions of genes in evolutionary processes. However, for a better understanding of this concept, it is highly 
recommended to read the [initial NEAT Paper](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf).

The NEAT settings that are used here are stored in the [config.txt](config.txt) file. Only the maximum Fitness threshold 
is changed directly in the [python script](NEAT_Pong.py) via the MAX_FITNESS variable.
### The Pong Neural Network
#### Input nodes
At each frame the input for the NN is the x and y position of the Ball in pixels as well as the current position of the 
paddle. 

#### Output node
The output is a single node with an tanh activation function. As this function pushes values in between -1 and 1. The 
decision of moving the paddle up or down is based on this output. If the output is below -0.5 the paddle moves down and 
if it is above 0.5 it is moved up. In between the paddle stays at its current height.

####  Fitness
For training it is necessary to create a fitness function that represents how well the neural network performs in its 
specific task. Here, the task is quite simple as it is only necessary to decide whether it is an good idea to move the
pong paddle either up or down at every frame. Therefore a specified parameter is added to the fitness if the AI 
successfully deflects a ball (LEARNING_PARAM1) or holds the paddle at the same height as the ball (LEARNING_PARAM2). 
As deflecting the ball is much more important than holding the paddle at the same height, it is a good idea to set the 
LEARNING_PARAM1 much higher than the other parameter. Additionally, the second parameter is added every frame, whilst 
the deflection parameter can only be added at the frame where the ball is deflected. 



## Training

The game and training parameters can be changed by editing the following global variables:

| Variable       | Effect        | 
| -------------- |:-------------| 
| WIN_WIDTH      | pygame window width in pixels | 
| WIN_HEIGHT     | pygame window height in pixels| 
| LEARNING_PARAM1| reward for the AI for deflecting a ball      | 
| LEARNING_PARAM2| reward for the AI for holding its paddle at the same height as the ball      | 
| MAX_Fitness    | maximum fitness threshold at which the AI is seen as perfect|
| FPS            | the number of FPS that pygame renders. This can be set very high to speed up training of the AI.|

Here it is possible to play around with the parameters and check how they influence the time that is necessary to
produce an unbeatable AI opponent. For example it is possible to set the reward for holding the paddle at the same 
height as the ball to zero, which might negatively influence the mean training time. 

The output of the program is a pickled neural network that can be included in a pygame pong game. This network is 
stored in the [winner.p](winner.p) file.






