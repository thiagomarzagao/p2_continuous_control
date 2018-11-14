### Result

The best result I managed to obtain was solving the taks in 183 episodes. The plot below shows how the scores as the episodes elapsed.

![scores X episodes](https://github.com/thiagomarzagao/p1_navigation/blob/master/Figure_1.png)

### The DDPG algorithm

To achieve that result I used an adapted version of the [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971) (DDPG) [code](https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py) provided in lesson 5. The main modification I made was to batch-normalize the output of the first hidden layer, after someone suggested it in the Student Hub. I also tweaked some of the hyperparameters (LR_ACTOR and LR_CRITIC). Finally, I set the size of each hidden layer at 128.

I tried other network archiectures (adding and subtracting hidden layers and changing the size of each hidden layer) but they didn't improve the model.

### Ideas for future improvements

In the future it might be worth it to try prioritized experience replay, as well as more recent approaches (like [Levine et al 2018](https://journals.sagepub.com/doi/abs/10.1177/0278364917710318)). Also, this took a while to run, so in the future I'd like to try using my laptop's GPU (right now that would be difficult because the latest CUDA realease is not yet compatible with XCode 10).
