# Mars Lander - Episode 2
# Genetic algorithm (GA).

__author__ = "Andrei Ermishin"
__copyright__ = "Copyright (c) 2019"
__license__ = "GNU GPLv3"
__maintainer__ = "Andrei Ermishin"
__email__ = "andrey.yermishin@gmail.com"

# The code couldn't pass Codingame test in time. 
# Version with 1 population class and list of chromosomes in it failed too.

import math
import random as rnd

# debug
import sys
import matplotlib.pyplot as plt


GRAVITY_MARS = 3.711
MIN_THRUST = 0
MAX_THRUST = 4
THRUST_STEP = 1
MIN_ANGLE = -90
MAX_ANGLE = 90
ANGLE_STEP = 15
SAFE_SPEED_X = 20 - 0.2
SAFE_SPEED_Y = 40 - 0.2
MAX_SPEED = 150 # coefficient for GA (high bound for free fall from 3000)
FLAT_GROUND = 1000
SURF_LEN_X = 7000
AREA_HEIGHT = 3000
ANGLE_RANGE = (-ANGLE_STEP, 0, ANGLE_STEP)
POWER_RANGE = (-THRUST_STEP, 0, THRUST_STEP)
POP_SIZE = 100
CHROM_SIZE = 100


class Chromosome:
    """ Chromosome is a list of genes/tuples of changes in (rotate, power). """
    def __init__(self, size, surface, center):
        self.x = 0
        self.y = 0
        self.x_y_path = []  # debug
        self.speed_x = 0
        self.speed_y = 0
        self.angle = 0
        self.power = 0
        self.total_power = 0
        self.fitness_val = 0
        self.surface = surface
        self.land_center = center
        self.intersect_idx = 0
        self.genes = [self.create_rnd_gene() for _ in range(size)]
    
    def intersect_surface(self):
        """ Return True if last (x, y) cross surface of Mars. """
        if self.x <= 0 or self.x >= SURF_LEN_X-1:
            return True
        
        x1, y1, x2, y2 = 0, 0, 0, 0
        for idx in range(len(self.surface) - 1):
            ### some nice peace of code here:
            only_author_has = 1
            real_code = only_author_has
        return self.y <= y_surf
    
    def update_coordinates(self, rotate, power):
        """ Compute coordinates (x, y) for every 1 second. """
        if MIN_ANGLE <= self.angle+rotate <= MAX_ANGLE:
            self.angle += rotate
        if MIN_THRUST <= self.power+power <= MAX_THRUST:
            self.power += power
        self.total_power += self.power
        
        # Vx = Vx0 + a*cos(ang)*t
        # Vy = Vy0 + (a*sin(ang) - g)*t
        # x = x0 + Vx*t + 1/2*a*cos(ang)*t*t
        # y = y0 + Vy*t + 1/2*(a*sin(ang) - g)*t*t
        # For movement within 1 second:
        angle_rad = math.radians(90 + self.angle)
        self.speed_x += self.power * math.cos(angle_rad)
        self.speed_y += self.power * math.sin(angle_rad) - GRAVITY_MARS
        self.x += self.speed_x + 1/2*self.power*math.cos(angle_rad)
        self.y += self.speed_y
        self.y += 1/2*(self.power*math.sin(angle_rad) - GRAVITY_MARS)

        self.x_y_path.append((self.x, self.y))  # debug
    
    def create_rnd_gene(self):
        """ Return tuple of random angle [-15, 0, 15] and power [-1, 0, 1]. """
        return rnd.choice(ANGLE_RANGE), rnd.choice(POWER_RANGE)
    
    def add_path(self, x, y, speed_x, speed_y, angle, power):
        """ Compute coordinates, angle, speed based on list of genes. """
        self.x = x
        self.y = y
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.angle = angle
        self.power = power
        
        self.x_y_path = [(x, y)]    # debug

        self.intersect_idx = 0
        self.total_power = 0
        for rotate, power in self.genes:
            self.update_coordinates(rotate, power)
            if self.intersect_surface():
                break
            self.intersect_idx += 1
        
        self.fitness_val = self.fitness()
    
    def fitness(self, k_dist=70, k_angle=1, k_v_x=12, k_v_y=12):
        """
        Return number between 0 (bad score) and 100 (good score).
        The closer a chromosome's (x, y) to landing center the better score.
        The smaller a chromosome's final speed and angle the better score.
        k_angle = 1 means that penalty range will be k_angle*[1...6].
        Sum of penalty points = 100.
        """
        max_score = 100
        penalty = 0
        # Distance to center of flat ground:
        distance = math.hypot(abs(self.land_center[0] - self.x),
                              abs(self.land_center[1] - self.y))
        # Chromosome ending at landing ground will have 70 points at least.
        distance = 0 if distance <= FLAT_GROUND/2 else distance
        penalty += k_dist * distance / (SURF_LEN_X - FLAT_GROUND/2)

        abs_angle = abs(self.angle)
        abs_speed_x = abs(self.speed_x)
        abs_speed_y = abs(self.speed_y)
        if (abs_angle > ANGLE_STEP
                or abs_speed_x > SAFE_SPEED_X
                or abs_speed_y > SAFE_SPEED_Y):
            # Angle penalty of k_angle*[1...6] points for [15...90] degrees.
            if abs_angle > ANGLE_STEP:
                penalty += k_angle * abs_angle / ANGLE_STEP
            
            # Speed x penalty of [1...k_v_x] points for [20...150] m/s.
            if abs_speed_x > SAFE_SPEED_X:
                extra_speed = abs_speed_x - SAFE_SPEED_X
                penalty += 1 + (k_v_x-1)*extra_speed/(MAX_SPEED - SAFE_SPEED_X)
            
            # Speed y penalty of [1...k_v_y] points for [40...150] m/s.
            if abs_speed_y > SAFE_SPEED_Y:
                extra_speed = abs_speed_y - SAFE_SPEED_Y
                penalty += 1 + (k_v_y-1)*extra_speed/(MAX_SPEED - SAFE_SPEED_Y)
        
        return max_score - penalty if max_score > penalty else 0
    
    # debug
    def __str__(self):
        return str(self.genes)


class Population:
    """ List of chromosomes with number of genes each. """
    def __init__(self, surface, size=POP_SIZE, genes=CHROM_SIZE):
        """ Make population of given size with random chromosomes. """
        self.x = 0
        self.y = 0
        self.speed_x = 0
        self.speed_y = 0
        self.angle = 0
        self.power = 0
        self.surface = surface
        center = self.get_land_center()
        self.chroms = [Chromosome(genes, surface, center) for _ in range(size)]
    
    def get_land_center(self):
        """ Return center (x, y) of landing spot if length >= FLAT_GROUND. """
        for idx in range(len(self.surface) - 1):
            pos1, pos2 = self.surface[idx], self.surface[idx + 1]
            if pos1[1] == pos2[1] and pos2[0]-pos1[0] >= FLAT_GROUND:
                return (pos1[0] + pos2[0])//2, pos1[1]
        return None
    
    def update_paths(self, x, y, speed_x, speed_y, angle, power):
        """ Update coordinates, speed, angle, power of chromosomes. """
        self.x = x
        self.y = y
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.angle = angle
        self.power = power

        for chrom in self.chroms:
            chrom.add_path(x, y, speed_x, speed_y, angle, power)

    def select_individuals(self):
        """
        Return a list of selected individuals from the population.
        Select the best 20% of individuals, then 20% of the rest randomly.
        """
        # Elitist selection:
        # Elitism allows best individuals to stay and guarantees 
        # that the solution quality obtained by the GA will not decrease 
        # from one generation to the next
        GRADED_PCT = 0.2    # % of retained best fitting individuals
        NONGRADED_PCT = 0.2 # % of retained remaining individuals (randomly)
        lst = sorted(self.chroms, key=lambda c: c.fitness_val)
        len_rest = int(len(lst) * (1-GRADED_PCT))
        graded = lst[len_rest:]
        
        # # a) Random selection:
        # non_graded = rnd.sample(lst[:len_rest], int(len_rest*NONGRADED_PCT))
        
        # b) Roulette wheel selection:
        # (The selections are made with probability according to the weights.)
        rest = lst[:len_rest]
        # rnd.choices([sequence], [weights], k)
        non_graded = rnd.choices(rest, [c.fitness_val for c in rest], 
                                 k=int(NONGRADED_PCT * len_rest))
        
        return graded + non_graded
    
    def clip_to_three(self, value, step):
        """ Clip the value to closest value from [-step, 0, step]. """
        if abs(value) < step/2:
            return 0
        elif value > 0:
            return step
        else:
            return -step
    
    def crossover_n_mutate(self, parent_1, parent_2, selected, children):
        """
        Return two children with new genes computed from parents.
        Mutate chromosome by a gene in [-15,15] for angle and [-1,1] for power.
        For chromosome representation with real numbers each gene 
        of the child chromosome has a chance to mutate.
        The crossover becomes a weighted sum of two chromosomes 
        using a random number between 0 and 1.
        """
        prob = 0.01  # probability of mutation : 1 %
        child_1 = None
        child_2 = None
        # Find two unselected childs:
        for chrom in self.chroms:
            if chrom not in selected and chrom not in children:
                if child_1 is None:
                    child_1 = chrom
                else:
                    child_2 = chrom
                    break
        
        genes = len(parent_1.genes)
        # # # 1) One point crossover:
        # # crossover_point = rnd.randrange(genes)
        # # 2) Mask crossover (gives more genetic diversity vs one point):
        # mask = [rnd.choice((0, 1)) for _ in range(genes)]
        for idx in range(genes):
        #     # # 1) One point crossover:
        #     # take_gene_from_same_parent = idx < crossover_point
        #     # 2) Mask crossover:
            ### some nice peace of code here:
            only_author_has = 1
            real_code = only_author_has
        
        return child_1, child_2
    
    def generate_next(self):
        """ Perform selection, crossover and mutation of individuals. """
        selected = self.select_individuals()
        childs_to_add = len(self.chroms) - len(selected)
        
        children = []
        # Add two children each time:
        while len(children) < childs_to_add-1:
            parent_1, parent_2 = rnd.sample(selected, 2)
            children.extend(self.crossover_n_mutate(parent_1, parent_2,
                                                    selected, children))
        
        # Update children chromosomes:
        for chrom in children:
            chrom.add_path(self.x, self.y, self.speed_x, self.speed_y,
                           self.angle, self.power)
    
    # debug
    def plot_x_y(self, pop_number=0):
        """ Plot list of (x, y) for each chromosome; plot surface. """
        for chrom in self.chroms:
            x_values, y_values = zip(*chrom.x_y_path)
            plt.plot(x_values, y_values)
        
        x_surf, y_surf = zip(*self.surface)
        plt.fill_between(x_surf, y_surf, 0, color='black', alpha=0.5)
        plt.title('Population({}) size = {}, num_genes= {}'.format(pop_number,
                                                    len(self.chroms),
                                                    len(self.chroms[0].genes)))
        plt.show()
    
    # debug
    def __str__(self):
        return '\n'.join(str(chrom) for chrom in self.chroms)


# HighGround:
surface = [
    (0, 1000), (300, 1500), (350, 1400), (500, 2100), (1500, 2100),
    (2000, 200), (2500, 500), (2900, 300), (3000, 200), (3200, 1000),
    (3500, 500), (3800, 800), (4000, 200), (4200, 800), (4800, 600),
    (5000, 1200), (5500, 900), (6000, 500), (6500, 300), (6999, 500)
]
# # DeepCanyon:
# surface = [
#     (0, 1000), (300, 1500), (350, 1400), (500, 2000), (800, 1800),
#     (1000, 2500), (1200, 2100), (1500, 2400), (2000, 1000), (2200, 500),
#     (2500, 100), (2900, 800), (3000, 500), (3200, 1000), (3500, 2000),
#     (3800, 800), (4000, 200), (5000, 200), (5500, 1500), (6999, 2800)
# ]
# surface = []
# # Points used to draw the surface of Mars. By linking all the points 
# # together in a sequential fashion, you form the surface of Mars.
# for i in range(int(input())):
#     # land_x: X coordinate of a surface point. (0 to 6999)
#     # land_y: Y coordinate of a surface point.
#     land_x, land_y = [int(j) for j in input().split()]
#     surface.append((land_x, land_y))
# # print('\n'.join(str(i) for i in surface), file=sys.stderr)

# Create the base population of chromosomes with random genes.
population = Population(surface)

solution = []
best_solution = None
gene_index = 0
first_turn = True
# game loop
while True:
    # h_speed: the horizontal speed (in m/s), can be negative.
    # v_speed: the vertical speed (in m/s), can be negative.
    # fuel: the quantity of remaining fuel in liters.
    # rotate: the rotation angle in degrees (-90 to 90).
    # power: the thrust power (0 to 4).
    if first_turn:
        # HighGround:
        x,y,h_speed,v_speed,fuel,rotate,power = 6500,2700, -50,0, 1000, 90, 0
        # # DeepCanyon:
        # x,y,h_speed,v_speed,fuel,rotate,power = 500,2700, 100,0, 800, -90, 0
    # x, y, h_speed, v_speed, fuel, rotate, power = map(int, input().split())


    # Find solution with GA at first turn:
    if first_turn:
        first_turn = False

        population.update_paths(x, y, h_speed, v_speed, rotate, power)
        population.plot_x_y()

        fitness_average = []
        count = 1
        while not solution and count < 1000:
            population.generate_next()
            
            if count == 10:
                population.plot_x_y(count)
            avg = sum(c.fitness_val for c in population.chroms)
            fitness_average.append(avg / len(population.chroms))

            for chromosome in population.chroms:
                if chromosome.fitness_val == 100:
                    solution.append(chromosome)
            count += 1
        
        if solution:
            population.plot_x_y(count)
            best_solution = solution[0]
            # Minimization of power usage:
            max_fuel = min(solution, key=lambda chrom: chrom.total_power)
            if max_fuel != best_solution:
                print('fuel consumption reduced by:',
                      best_solution.total_power-max_fuel.total_power,
                      file=sys.stderr)
                best_solution = max_fuel
            
            print('best_solution speed:', best_solution.speed_x,
                                          best_solution.speed_y)
        else:
            best_solution = max(population.chroms, key=lambda c: c.fitness_val)
            print('No solution found!', file=sys.stderr)
            print('The best for now (x, y, fitness):', best_solution.x,
                  best_solution.y, best_solution.fitness_val, file=sys.stderr)
        
        plt.plot(fitness_average)
        plt.title(f'Fitness average(population size={POP_SIZE})')
        plt.show()
    
    # Print genes from solution (chromosome):
    if gene_index < len(best_solution.genes):
        add_angle, add_power = best_solution.genes[gene_index]
        if MIN_ANGLE <= rotate+add_angle <= MAX_ANGLE:
            rotate += add_angle
        if MIN_THRUST <= power+add_power <= MAX_THRUST:
            power += add_power
        # Force output: angle=0 for last two genes.
        if gene_index >= best_solution.intersect_idx-2:
            rotate = 0
    
    print(rotate, power)
    gene_index += 1

    if not first_turn:
        break
