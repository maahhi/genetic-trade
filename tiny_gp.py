# tiny genetic programming by © moshe sipper, www.moshesipper.com
from random import random, randint, seed
from statistics import mean
from copy import deepcopy
import math
import pandas as pd

POP_SIZE = 60  # population size
MIN_DEPTH = 2  # minimal initial random tree depth
MAX_DEPTH = 5  # maximal initial random tree depth
GENERATIONS = 250  # maximal number of generations to run evolution
TOURNAMENT_SIZE = 5  # size of tournament for tournament selection
XO_RATE = 0.8  # crossover rate
PROB_MUTATION = 0.2  # per-node mutation probability


class Functions:

    def __init__(self, price_series, volume_series, time):
        self.price = price_series
        self.volume = volume_series
        self.current_time = time

    @staticmethod
    def add(x, y): return x + y

    @staticmethod
    def sub(x, y): return x - y

    @staticmethod
    def norm(x, y): return abs(x-y)

    @staticmethod
    def mul(x, y): return x * y

    @staticmethod
    def div(x, y):
        if y!= 0:
            return x / y
        else:
            return math.inf

    @staticmethod
    def and_f(x, y): return x and y

    @staticmethod
    def or_f(x, y): return x or y

    @staticmethod
    def not_f(x): return not x

    @staticmethod
    def larger(x, y): return x > y

    @staticmethod
    def smaller(x, y): return x < y

    @staticmethod
    def if_then_else(x, y, z):
        if x:
            return y
        else:
            return z

    def average(self, p_v, n):
        if p_v:
            return self.price[self.current_time - n, self.current_time] / n
        else:
            return self.volume[self.current_time - n, self.current_time] / n

    def max(self, p_v, n):
        if p_v:
            return max(self.price[self.current_time - n, self.current_time])
        else:
            return max(self.volume[self.current_time - n, self.current_time])

    def min(self, p_v, n):
        if p_v:
            return min(self.price[self.current_time - n, self.current_time])
        else:
            return min(self.volume[self.current_time - n, self.current_time])

    def lag(self, p_v, n):
        if p_v:
            return self.price[self.current_time - n]
        else:
            return self.volume[self.current_time - n]

    def volatility(self, n):
        avg = sum(self.price) / len(self.price)
        return sum((x - avg) ** 2 for x in self.price) / len(self.price)


# import data
df = pd.read_csv('BINANCE_ETHBTC_H4.csv')
price_pseries = df.open
volume_pseries = df.volume

func = Functions(price_pseries, volume_pseries)
arithmetic_funcs = [func.add, func.sub, func.mul, func.div, func.norm]
boolean_funcs = [func.and_f, func.or_f]
number_to_boolean_funcs = [func.larger, func.smaller]
price_volume_funcs = [func.min, func.max, func.lag, func.average]
FUNCTIONS = arithmetic_funcs + boolean_funcs + number_to_boolean_funcs + price_volume_funcs

boolean_terms = [True,False]
n_terms = [2,6,12,18,42,360,720]# 8h, 1d, 2d, 3d, 7d, 30d, 60d
random_terms =[-2, -1, 0, 1, 2]
TERMINALS = boolean_terms + n_terms + random_terms


def target_func(x):  # evolution's target
    return x * x * x * x + x * x * x + x * x + x + 1


def generate_dataset():  # generate 101 data points from target_func
    dataset = []
    for x in range(-100, 101, 2):
        x /= 100
        dataset.append([x, target_func(x)])
    return dataset


class GPTree:
    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    def node_label(self):  # string label
        if (self.data in FUNCTIONS):
            return self.data.__name__
        else:
            return str(self.data)

    def print_tree(self, prefix=""):  # textual printout
        print("%s%s" % (prefix, self.node_label()))
        if self.left:  self.left.print_tree(prefix + "   ")
        if self.right: self.right.print_tree(prefix + "   ")

    def compute_tree(self, x):
        if (self.data in FUNCTIONS):
            return self.data(self.left.compute_tree(x), self.right.compute_tree(x))
        elif self.data == 'x':
            return x
        else:
            return self.data

    def random_tree(self, grow, max_depth, depth=0):  # create random tree using either grow or full method
        if depth < MIN_DEPTH or (depth < max_depth and not grow):
            self.data = FUNCTIONS[randint(0, len(FUNCTIONS) - 1)]
        elif depth >= max_depth:
            self.data = TERMINALS[randint(0, len(TERMINALS) - 1)]
        else:  # intermediate depth, grow
            if random() > 0.5:
                self.data = TERMINALS[randint(0, len(TERMINALS) - 1)]
            else:
                self.data = FUNCTIONS[randint(0, len(FUNCTIONS) - 1)]
        if self.data in FUNCTIONS:
            self.left = GPTree()
            self.left.random_tree(grow, max_depth, depth=depth + 1)
            self.right = GPTree()
            self.right.random_tree(grow, max_depth, depth=depth + 1)

    def mutation(self):
        if random() < PROB_MUTATION:  # mutate at this node
            self.random_tree(grow=True, max_depth=2)
        elif self.left:
            self.left.mutation()
        elif self.right:
            self.right.mutation()

    def size(self):  # tree size in nodes
        if self.data in TERMINALS: return 1
        l = self.left.size() if self.left else 0
        r = self.right.size() if self.right else 0
        return 1 + l + r

    def build_subtree(self):  # count is list in order to pass "by reference"
        t = GPTree()
        t.data = self.data
        if self.left:  t.left = self.left.build_subtree()
        if self.right: t.right = self.right.build_subtree()
        return t

    def scan_tree(self, count, second):  # note: count is list, so it's passed "by reference"
        count[0] -= 1
        if count[0] <= 1:
            if not second:  # return subtree rooted here
                return self.build_subtree()
            else:  # glue subtree here
                self.data = second.data
                self.left = second.left
                self.right = second.right
        else:
            ret = None
            if self.left and count[0] > 1: ret = self.left.scan_tree(count, second)
            if self.right and count[0] > 1: ret = self.right.scan_tree(count, second)
            return ret

    def crossover(self, other):  # xo 2 trees at random nodes
        if random() < XO_RATE:
            second = other.scan_tree([randint(1, other.size())], None)  # 2nd random subtree
            self.scan_tree([randint(1, self.size())], second)  # 2nd subtree "glued" inside 1st tree


# end class GPTree

def init_population():  # ramped half-and-half
    pop = []
    for md in range(3, MAX_DEPTH + 1):
        for i in range(int(POP_SIZE / 6)):
            t = GPTree()
            t.random_tree(grow=True, max_depth=md)  # grow
            pop.append(t)
        for i in range(int(POP_SIZE / 6)):
            t = GPTree()
            t.random_tree(grow=False, max_depth=md)  # full
            pop.append(t)
    return pop


def fitness(individual, dataset):  # inverse mean absolute error over dataset normalized to [0,1]
    return 1 / (1 + mean([abs(individual.compute_tree(ds[0]) - ds[1]) for ds in dataset]))


def selection(population, fitnesses):  # select one individual using tournament selection
    tournament = [randint(0, len(population) - 1) for i in range(TOURNAMENT_SIZE)]  # select tournament contenders
    tournament_fitnesses = [fitnesses[tournament[i]] for i in range(TOURNAMENT_SIZE)]
    return deepcopy(population[tournament[tournament_fitnesses.index(max(tournament_fitnesses))]])


def main():
    # init stuff
    seed()  # init internal state of random number generator
    dataset = generate_dataset()
    population = init_population()
    best_of_run = None
    best_of_run_f = 0
    best_of_run_gen = 0
    fitnesses = [fitness(population[i], dataset) for i in range(POP_SIZE)]

    # go evolution!
    for gen in range(GENERATIONS):
        nextgen_population = []
        for i in range(POP_SIZE):
            # question: where the previous generation gone? did he throw them away ????
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)
            parent1.crossover(parent2)
            parent1.mutation()
            nextgen_population.append(parent1)
        population = nextgen_population
        fitnesses = [fitness(population[i], dataset) for i in range(POP_SIZE)]
        if max(fitnesses) > best_of_run_f:
            best_of_run_f = max(fitnesses)
            best_of_run_gen = gen
            best_of_run = deepcopy(population[fitnesses.index(max(fitnesses))])
            print("________________________")
            print("gen:", gen, ", best_of_run_f:", round(max(fitnesses), 3), ", best_of_run:")
            best_of_run.print_tree()
        if best_of_run_f == 1: break

    print("\n\n_________________________________________________\nEND OF RUN\nbest_of_run attained at gen " + str(
        best_of_run_gen) + \
          " and has f=" + str(round(best_of_run_f, 3)))
    best_of_run.print_tree()


if __name__ == "__main__":
    main()

