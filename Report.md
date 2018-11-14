### Task

The goal is to create an agent that learns how to catch a moving target. Every time step that the agent's double-jointed arm is on the target the agent gets a reward of +0.1.

The state space has 33 dimensions (position, rotation, velocity, and agular velocities of the arm). Given this information, the agent has to learn how to best select actions. The action space has 4 dimensions; each entry corresponds to the torque of one joint of one arm and can vary from -1 to +1.

The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

### Result

The best result I managed to obtain was solving the taks in 183 episodes. The plot below shows how the scores as the episodes elapsed.

![scores X episodes](https://github.com/thiagomarzagao/p2_continuous_control/blob/master/Figure_1.png)

### The DDPG algorithm

To achieve that result I used an adapted version of the [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971) (DDPG) [code](https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py) provided in lesson 5. The DDPG algorithm is similar to the [DQN](https://www.nature.com/articles/nature14236) algorithm in that there are two neural networks with identical architecture. But DQN can't handle continuous action spaces - which is the case here, as each of the four dimensions of the action space ranges from -1 to +1. The DDPG algorithm handles that by having one neural net - the actor - pick (what it believes to be) the best policy for each state and having the other neural net - the critic - evaluate those policy choices. That way we are able to work with continuous spaces. DDPG also uses replay buffer, just like DQN, but with "soft" updates - we don't outright clone one network into the other; instead we make them closer to each other at each update. (Though as we saw in the previous module it's possible to use soft updates with the DQN algorithm too.)

The structure of both my neural nets (actor and critic) is the same: one input layer of size 33 (the size of the state space), two hidden layers of size 128 each, and one output layer of size 4 (the size of the action space). The activation function is ReLU except for the output layer, where I used tanh (because each of the four action parameters had to be between -1 and +1). Following a comment I saw on Student Hub I batch-normalized the output of the first hidden layer - that greatly improved the result (before doing that I was stuck in an average score of ~5 even after over 1k episodes). I also tweaked some of the hyperparameters (LR_ACTOR and LR_CRITIC).

I tried other network architectures (adding and subtracting hidden layers and changing the size of each hidden layer) but they didn't improve the model.

### Ideas for future improvements

In the future it might be worth it to try prioritized experience replay, as well as more recent approaches (like [Levine et al 2018](https://journals.sagepub.com/doi/abs/10.1177/0278364917710318)). Also, this took a while to run, so in the future I'd like to try using my laptop's GPU (right now that would be difficult because the latest CUDA realease is not yet compatible with XCode 10).
