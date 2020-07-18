import pygame
import random
import os
import neat
WIN_WIDTH = 1000
WIN_HEIGHT = 800
WHITE =     (255, 255, 255)
BLUE =      (  0,   0, 255)
GREEN =     (  0, 255,   0)
RED =       (255,   0,   0)
BLACK =     (  0,   0,   0)


# The Player class is used to initiate objects of Pong paddles
# init:
#   y:          starting height of the Player
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


class Ball:
    def __init__(self, x, y, radius, x_vel, y_vel):
        self.x = x
        self.y = y
        self.x_vel = round(x_vel)
        self.y_vel = random.choice([-1, 1]) * (random.randint(2, x_vel*10)/10)
        self.radius = radius
        self.fit = 0

    def move(self):
        if (self.y + self.y_vel) - self.radius < 0 or self.y + self.y_vel + self.radius > WIN_HEIGHT:
            self.y_vel = -self.y_vel

        self.y = self.y + self.y_vel
        self.x = self.x + self.x_vel


    def draw(self, win):
        pygame.draw.circle(win, BLACK, [round(self.x), round(self.y)], self.radius)

    def colision(self, player):
        if player.side == "left":
            if (self.x + self.x_vel) - self.radius < 0 + player.width:
                if self.y < player.y + player.height and self.y > player.y:
                    self.x_vel = -(self.x_vel + 0.5)
                    self.y_vel = self.y_vel + random.choice([-1, 1]) * random.randint(0, 10)/10
                    return 1

        if player. side == "right":
            if (self.x + self.x_vel) + self.radius > WIN_WIDTH - player.width:
                if self.y < player.y + player.height and self.y > player.y:
                    self.x_vel = -(self.x_vel + 0.5)
                    self.y_vel = self.y_vel + random.choice([-1, 1]) * random.randint(0, 10)/10
        return 0
    def scored(self):
        if (self.x + self.x_vel) - self.radius < 0:
            return "right"
        elif(self.x + self.x_vel) + self.radius > WIN_WIDTH:
            return "left"
        else:
            return False

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



def main(genomes, config):
    nets = []
    ge = []
    for gid, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        ge.append(genome)
    for x, genome in enumerate(ge):

        pygame.font.init()
        font = pygame.font.SysFont('Consolas', 30)
        score = {"left": 0,
                 "right": 0}
        bg = pygame.Rect(0, 0, WIN_WIDTH, WIN_HEIGHT)
        win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        players = [Player(round(WIN_HEIGHT/2 - 200/2), 20, 200, "left"),
                   Player(round(WIN_HEIGHT/2 - WIN_HEIGHT/2), 20, WIN_HEIGHT, "right")]
        win.fill([255, 0, 255])
        pygame.display.flip()
        ball = Ball(int(WIN_WIDTH/2), int(WIN_HEIGHT/2), 10, 3, -3)
        clock = pygame.time.Clock()
        run = True
        while run:
            clock.tick(1000)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False


            for player in players:
                ge[x].fitness += ball.colision(player)
                ball.colision(player)
            ball.move()
            output = nets[x].activate((ball.x, ball.y, players[0].y))
            if output[0] < -0.5:
                players[0].move_down()
            elif output[0] > 0.5:
                players[0].move_up()
            else:
                pass

            draw_window(win, ball, bg, players, font, score)
            if ball.scored() is not False:
                run = False


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)


    # displays the best genome including the best performing neural network. A further step would be pickeling
    # this network and including it as an AI opponent in a Pong game that can be played by a human player.
    winner = p.run(main, 200)
    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run(config_path=config_path)


