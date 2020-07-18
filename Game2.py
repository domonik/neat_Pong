# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Name:         NEAT-Pong
Beschreibung: training of an AI-Pong opponent using pythons integration of NEAT
              http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
Autor:        Dominik Rabsch
Datum:        18.07..2020
-------------------------------------------------------------------------------
"""


import pygame
import random
import os
import neat
WIN_WIDTH = 1000  # pygame window width in pixels
WIN_HEIGHT = 800  # pygame window height in pixels
LEARNING_PARAM1 = 1  # reward for the AI for deflecting a ball
LEARNING_PARAM2 = 0.001  # reward for the AI for holding its paddle at the same height as the ball
MAX_FITNESS = 150  # maximum fitness threshold at which the AI is seen as perfect
FPS = 500  # defines the number of FPS that pygame renders. This can be set very high to speed up training of the AI.

# tuples of RGB values that are used a few times
WHITE =     (255, 255, 255)
BLACK =     (  0,   0,   0)


# The Player class is used to initiate objects of Pong paddles
# init:
#   y:          starting y position the Player
#   height:     height of the Players paddle
#   width:      width of the Players paddle
#   side:       specifies whether the paddle is on the left or right side of the field
# methods:
#   move_up:    moves the paddle up at a specific speed
#   move_down:  moves the paddle down at the specified speed
#   draw:       draws the paddle into the pygame window (win)
class Player:

    speed = 3  # number of pixels the paddle can move each frame

    def __init__(self, y, width, height, side="left"):
        self.side = side
        self.y = y  #
        self.height = height
        self.width = width

        if self.side == "right":  # sets the paddle to the right
            self.x = WIN_WIDTH - self.width
        elif self.side == "left":  # sets the paddle to the left
            self.x = 0
        else:
            raise Exception("side must be either left or right")
        self.rect = pygame.Rect(self.x, self.y, width, height)

    def move_up(self):
        if self.y > 0:  # prevents from moving out of bounds
            self.y = self.y - self.speed
            self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def move_down(self):
        if self.y + self.height < WIN_HEIGHT:  # prevents from moving out of bounds
            self.y = self.y + self.speed
            self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def draw(self, win):
        pygame.draw.rect(win, BLACK, self.rect, 0)


# The Ball class is used to initiate objects of a Ball
# init:
#   x:          starting x position of the ball
#   y:          starting y position of the ball
#   radius:     radius of the ball in px
#   x_vel:      speed in x direction (px/frame) of the ball
#
# methods:
#   move:       moves the ball with its current velocity (x_vel, y_vel)
#   draw:       draws the ball at its current position (x,y) into the pygame window (win)
#   collision:  detects whether a player successfully deflects the ball
#   scored:     returns left or right depending on which side has scored a point
class Ball:
    def __init__(self, x, y, radius, x_vel):
        self.x = x
        self.y = y
        self.x_vel = round(x_vel)
        # sets random y velocity when the Ball is created
        self.y_vel = random.choice([-1, 1]) * (random.randint(2, x_vel*10)/10)
        self.radius = radius

    def move(self):
        # deflection at the y boundaries
        if (self.y + self.y_vel) - self.radius < 0 or self.y + self.y_vel + self.radius > WIN_HEIGHT:
            self.y_vel = -self.y_vel

        self.y = self.y + self.y_vel
        self.x = self.x + self.x_vel

    def draw(self, win):
        pygame.draw.circle(win, BLACK, [round(self.x), round(self.y)], self.radius)

    # The return value of this function is not important for a normal Pong game. However, in this case it is used to
    # increase the fitness of a neural network (fitness is explained later in this file). The fitness increases if the
    # AI either successfully deflects the ball or has its paddle at the same height as the ball.
    def collision(self, player):
        ret_value = 0
        if player.side == "left":
            if player.y + player.height > self.y > player.y:
                ret_value += LEARNING_PARAM2  # adds the same height learning parameter to the return value
                if (self.x + self.x_vel) - self.radius < 0 + player.width:
                    self.x_vel = -(self.x_vel + 0.5)  # speeds up the ball for every deflection
                    # adds some random deflection to prevent the ball from getting stuck at the same route
                    self.y_vel = self.y_vel + random.choice([-1, 1]) * random.randint(0, 10)/10
                    ret_value += LEARNING_PARAM1  # adds the deflection learning parameter to the return value

        if player. side == "right":
            if (self.x + self.x_vel) + self.radius > WIN_WIDTH - player.width:
                if player.y + player.height > self.y > player.y:
                    self.x_vel = -(self.x_vel + 0.5)  # speeds up the ball for every deflection
                    # adds some random deflection to prevent the ball from getting stuck at the same route
                    self.y_vel = self.y_vel + random.choice([-1, 1]) * random.randint(0, 10)/10
        # prevents the ball from getting too fast.
        if self.y_vel > Player.speed:
            self.y_vel = Player.speed
        elif self.y_vel < - Player.speed:
            self.y_vel = - Player.speed

        return ret_value

    # this function is not needed during training of the neural network as the game will end as soon as the AI fails to
    # deflect the ball. However, it can be used later when playing against an already trained AI opponent.
    def scored(self):
        if (self.x + self.x_vel) - self.radius < 0:
            return "right"
        elif(self.x + self.x_vel) + self.radius > WIN_WIDTH:
            return "left"
        else:
            return False


# draws players, the ball, the score and the background into a specified pygame window. During training the score of
# 0:0 wont change as as the game will end as soon as the AI fails to deflect the ball. However, it can be used later
# when playing against an already trained AI opponent.
# arguments:
#   win:        the pygame window in which everything should be drawn
#   ball:       the ball that is drawn into the pygame window
#   bg:         the background rectangle of the pong field
#   players:    a list of players that is drawn into the window
#   font:       font of the score
#   score:      dictionary with keywords left and right and the corresponding points scored
def draw_window(win, ball, bg, players, font, score):
    pygame.draw.rect(win, WHITE, bg, 0)
    leftscore = font.render( "left:  " + str(score["left"]), False, (0, 0, 0))
    rightscore = font.render("right: " + str(score["right"]), False, (0, 0, 0))
    rightsize = rightscore.get_size()
    leftsize = leftscore.get_size()
    win.blit(leftscore, (WIN_WIDTH/2 - rightsize[0]/2, 0))
    win.blit(rightscore, (WIN_WIDTH/2 - leftsize[0]/2, leftsize[1]))
    for player in players:
        player.draw(win)
    ball.draw(win)
    pygame.display.update()

# The main function of the game. It takes a list of genomes as well as a config object for these genomes.
# Both of them are provided by the neat pythons populations run function. For each genome a slightly altered neural
# network is created based on the settings provided by the config object. Every genome object has a variable called
# fitness. This fitness is a measure of how well the network performs in playing Pong. In this case the fitness is
# increased if either the AI successfully deflects a ball or holds its paddle at the same height as the ball.
# The Neural network has 3 ingoing values. Namely the y position of the paddle as well as the x and y position of the
# ball. The output is a single node with an tanh activation function. As the functions values are between -1 and 1,
# the AI moves the paddle up if the output value is higher than 0.5 or down if it is below -0.5.
def main(genomes, config):
    nets = []
    ge = []
    # creates Feed Forward networks for each genome in a population
    for gid, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        ge.append(genome)

    # playes Pong for each genome
    for x, genome in enumerate(ge):

        pygame.font.init()
        font = pygame.font.SysFont('Consolas', 30)
        score = {"left": 0,
                 "right": 0}
        bg = pygame.Rect(0, 0, WIN_WIDTH, WIN_HEIGHT)
        win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        # the right side player is set to be of the same size as the windows height as this will mirror an opponent
        # that deflects all balls. This is necessary during training of the neural network
        players = [Player(round(WIN_HEIGHT/2 - 200/2), 20, 200, "left"),
                   Player(round(WIN_HEIGHT/2 - WIN_HEIGHT/2), 20, WIN_HEIGHT, "right")]
        win.fill([255, 0, 255])
        pygame.display.flip()
        ball = Ball(int(WIN_WIDTH/2), int(WIN_HEIGHT/2), 10, 3)
        clock = pygame.time.Clock()
        game_alive = True

        # the Pong game will end as soon as the Network misses a ball or it reaches the maximum fitness threshold
        while game_alive:
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False


            for player in players:
                ge[x].fitness += ball.collision(player)

            # Ends the game if the fitness reaches a maximum value. Else the game continues forever as the AI wont
            # make any mistakes.
            if ge[x].fitness >= MAX_FITNESS:
                game_alive = False
            ball.move()

            output = nets[x].activate((ball.x, ball.y, players[0].y))
            # the output of the neural network decides whether the paddle is moved up/down or stays at its current
            # position
            if output[0] < -0.5:
                players[0].move_down()
            elif output[0] > 0.5:
                players[0].move_up()
            else:
                pass

            draw_window(win, ball, bg, players, font, score)
            # end the game as soon as the AI misses to deflect the ball
            if ball.scored() is not False:
                game_alive = False


def run_training(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_path)

    # overrides the fitness threshold of the config file to make it accessible as a global Variable in this script.
    config.fitness_threshold = MAX_FITNESS

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)


    # displays the best genome including the best performing neural network. A further step would be pickeling
    # this network and including it as an AI opponent in a Pong game that can be played by a human player.
    # As the problem that needs to be solved is quite simple, the best network might consist of only a few (~ 3) nodes.
    winner = p.run(main, 200)
    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run_training(config_path=config_path)


