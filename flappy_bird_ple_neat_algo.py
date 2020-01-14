from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import neat
import os
import numpy as np
import pickle
import time

game = FlappyBird()
ple = PLE(game, fps=30, display_screen=True)
ple.init()

action_set = ple.getActionSet()
ple.reset_game()


# {'player_y': 54.0, 'player_vel': 1.0, 'next_pipe_dist_to_player': 165.0, 'next_pipe_top_y': 147, 'next_pipe_bottom_y': 247, 'next_next_pipe_dist_to_player': 309.0, 'next_next_pipe_top_y': 43, 'next_next_pipe_bottom_y': 143}

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        ple.reset_game()

        action = action_set[random.randint(0, 1)]
        ple.act(action)
        fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        distance_total = 0
        game_over = False

        while not game_over:
            observation = ple.getGameState()
            data = [observation.get('player_y'), observation.get('next_pipe_top_y'),
                    observation.get('next_pipe_bottom_y'),observation.get('next_pipe_dist_to_player'),
                    observation.get('player_vel')
                    ]
            output = np.argmax(net.activate(data))
            reward = ple.act(action_set[output])

            if reward > 0:
                fitness += 5
                print(fitness)
            genome.fitness = fitness
            if ple.game_over():
                game_over = True


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    start = time.time()
    winner = p.run(eval_genomes, 50)
    end = time.time()
    print(str(end - start) + ' seconds')
    print('win')
    pickle_file = open('winner.pickle', 'wb')
    pickle.dump(winner, pickle_file)
    pickle_file.close()
    best_flappybird_run(config)



def best_flappybird_run(config):
    game = FlappyBird()
    ple = PLE(game, fps=30, display_screen=True)
    ple.getScreenRGB()
    ple.init()
    pickle_file = open('winner.pickle', 'rb')
    winner = pickle.load(pickle_file)
    net = neat.nn.FeedForwardNetwork.create(winner,config)
    pickle_file.close()
    done = False
    ple.reset_game()

    while not done:
        observation = ple.getGameState()
        data = [observation.get('player_y'), observation.get('next_pipe_top_y'),
                observation.get('next_pipe_bottom_y'), observation.get('next_pipe_dist_to_player'),
                observation.get('player_vel')
                ]
        output = np.argmax(net.activate(data))
        ple.act(action_set[output])

        if ple.game_over():
            done = True


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    """   config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    start = time.time()
    best_flappybird_run(config)
    end = time.time()
    print(str(end - start) + ' seconds')"""
    run(config_path)
