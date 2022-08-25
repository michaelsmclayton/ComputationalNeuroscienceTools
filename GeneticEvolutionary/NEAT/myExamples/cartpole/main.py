# --------------------
# Define game
# --------------------
# requires pip3 install gym==0.21.0
import gym
env = gym.make("CartPole-v1")

# Observation and action space 
obs_space = env.observation_space
action_space = env.action_space
print("The observation space: {}".format(obs_space))
print("The action space: {}".format(action_space))

# --------------------
# Define NEAT optimiser
# --------------------
import neat
import numpy as np

# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feedforward')

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(False)) # add a stdout reporter to show progress in the terminal.

# --------------------
# Define network evaluation functions
# --------------------

# Evaluation genome
def evaluate(env, net, timepoints, render=False):
    obs = env.reset()
    rewards = np.zeros(timepoints)
    for t in range(timepoints):
        actions = net.activate(obs)
        # actions = np.random.choice([0,1])
        actions = int(np.round(actions))
        obs, reward, done, info = env.step(actions)
        rewards[t] = reward
        if render==True:
            env.render()
        if np.abs(obs[0])>4.8:
            return t
    return t*np.sum(rewards)

def eval_genomes(genomes, config):
    global env
    for genome_id, genome in genomes:
        net = neat.nn.RecurrentNetwork.create(genome, config)
        genome.fitness = evaluate(env, net, 300)

# --------------------
# Run evolution!
# --------------------

# Run until a solution is found.
winner = p.run(eval_genomes, 100)

# --------------------
# Return best genome
# --------------------

# Run the winning genome.
best_genome = p.best_genome
net = neat.nn.RecurrentNetwork.create(best_genome, config)
r = evaluate(env, net, 300, render=True)

# Draw net
import visualize
d = visualize.draw_net(config, best_genome)