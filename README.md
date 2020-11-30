## Data
Dataset consists of 1k+ features, timestamp, PUE value for the next timestamp (label).

Features represent some measurements from the installed equipment (temperature, humidity etc).
~~First 64 values can be adjusted.~~


## Feature selection
Firstly only some features are selected for the following models. 
Data preparation can be done using `feature_selection` function, which returns 
first 64 features + any other number of features that you may want to pick.
Features are selected by their importance (first n most important). 
Importance is calculated with SelectKBest, selection algorithm can be specified. 
You can also apply dimensionality reduction to the not-first-64 features with PCA.

`feature_selection` returns 3 numpy arrays: features, timestamp, label.


## LSTM model (Environment)
Firstly, LSTM-based model is created to predict PUE value. Because the model is trained on 
sequential batches of training data, so LSTM layers has `stateful=True` flag. It's done in
order to give model an understanding of data's sequential nature without passing the timestamp.

The model returns predicted value(PUE value for the next timestamp).

Then this model is wrapped by `Environment` class, that is later used in RL part.
`Environment` has `step` function that takes values for each adjustable equipment, and returns
a new state (a new current PUE value) and reward (difference between the new PUE with 
and an ideal value == 1.0).


## Actor-Critic model (RL agent)
For the environment optimization purposes a variation of vanilla Actor-Critic is implemented.
[Google's Actor-Critic model for playing CartPole](https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic) 
was used as a reference, while creating this model. Agent has unified input (state): 
64 + N + PUE (for the current timestamp). And separated outputs for actor (action) and critic parts(expected reward).
Critic tries to predict environment's reward(return) value, and actor generates action (64 + N values),
that should be taken next in the environment. Opposed to the traditional approach when actor returns
a probability of taking a certain action, this model gives you direct values for the *equipment* in the environment.
And because of this straightforwardness it's quite hard to train agent properly. The most complicated 
thing here would be: how to communicate back to the agent whether it found an optimal solution?

### Loss function
Loss function is quite simple actually. Because we want for critic to make good predictions of 
environment reward on certain actions, and for actor to perform well(to get the highest rewards/returns)
in the environment, loss function should take into account both of these terms. 

When it comes to evaluating critic performance it's quite easy we just have to calculate a difference between
actual environment's reward/return and agent's one. Different ways to do that can be used, here the Huber Loss
function is used. 

For the actor it's a bit more complicated, as this way of using Actor-Critic method very isn't common 
(i.e. optimizing on the environment with straightaway values), and there's no definite way to say
if one of the parts of the action(i.e. value for *ith* equipment) is more important than another. 
So I came up with a pretty crude way of transmitting this back to the model, just using received from
environment reward/reward as a part of the loss function.

So loss function consists of two terms: 
```python
    critic_loss + returns_sum
```
where `returns_sum` is an absolute sum of rewards/returns for the whole episode. It's alright to minimize 
rewards in this case, because in this environment returns are (almost) never positive.