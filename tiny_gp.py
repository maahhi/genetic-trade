# tiny genetic programming by © moshe sipper, www.moshesipper.com
from random import random, randint, seed
from statistics import mean
from copy import deepcopy
from Functions import Functions
import pandas as pd

POP_SIZE = 60  # population size
MIN_DEPTH = 2  # minimal initial random tree depth
MAX_DEPTH = 5  # maximal initial random tree depth
GENERATIONS = 250  # maximal number of generations to run evolution
TOURNAMENT_SIZE = 5  # size of tournament for tournament selection
XO_RATE = 0.8  # crossover rate
PROB_MUTATION = 0.2  # per-node mutation probability


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
FUNCS_info = {'add':['n','n','n'],'sub':['n','n', 'n'],'mul':['n','n','n'],'div':['n','n','n'],'norm':['n','n','n'],
              'and_f':['b','b','b'],'or_f':['b','b','b'],
              'larger':['b','n','n'],'smaller':['b','n','n'],
              'min':['n','b','nn'],'max':['n','b','nn'],'lag':['n','b','nn'],'average':['n','b','nn'],}
boolean_terms = [True,False]
n_terms = [2,6,12,18,42,360,720]# 8h, 1d, 2d, 3d, 7d, 30d, 60d
random_terms =[-2, -1, -0.2, -0.5, -0.8, 0, 0.2, 0.5, 0.8, 1, 2]
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

        if data in FUNCTIONS:
            self.left_type = FUNCS_info[data.__name__][1]
            self.right_type = FUNCS_info[data.__name__][2]
            self.output_type = FUNCS_info[data.__name__][0]
        if data in TERMINALS:
            self.right_type = None
            self.left_type = None
            self.output_type = None

    def node_label(self):  # string label
        if (self.data in FUNCTIONS):
            return self.data.__name__
        else:
            return str(self.data)

    def print_tree(self, prefix=""):  # textual printout
        print("%s%s" % (prefix, self.node_label()))
        if self.left:  self.left.print_tree(prefix + "   ")
        if self.right: self.right.print_tree(prefix + "   ")

    def compute_tree(self):

        if (self.data in FUNCTIONS):
            return self.data(self.left.compute_tree(), self.right.compute_tree())
        else:
            return self.data

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
            if not second:  # return subtree rooted here --None--
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

        other
        if random() < XO_RATE:
            second = other.scan_tree([randint(1, other.size())], None)  # 2nd random subtree
            second.output_type
            self.scan_tree([randint(1, self.size())], second)  # 2nd subtree "glued" inside 1st tree

    def tree_constructor(self, depth):
        if self.right_type == 'b':
            if depth + 1 >= MAX_DEPTH or random() < 0.3:
                possible_terminals = boolean_terms
                self.right = GPTree(possible_terminals[randint(0, len(possible_terminals) - 1)])
            else:
                possible_functions = boolean_funcs + number_to_boolean_funcs
                self.right = GPTree(possible_functions[randint(0, len(possible_functions) - 1)])
                self.right.tree_constructor(depth + 1)
        elif self.right_type == 'n':
            if depth + 1 >= MAX_DEPTH or random() < 0.3:
                possible_terminals = random_terms
                self.right = GPTree(possible_terminals[randint(0, len(possible_terminals) - 1)])
            else:
                possible_functions = arithmetic_funcs + price_volume_funcs + price_volume_funcs
                self.right = GPTree(possible_functions[randint(0, len(possible_functions) - 1)])
                self.right.tree_constructor(depth + 1)
        elif self.right_type == 'nn':
            possible_terminals = n_terms
            self.right = GPTree(possible_terminals[randint(0, len(possible_terminals) - 1)])
        else:
            print("in tree constructor, right life is None ! ")
        if self.left_type == 'b':
            if not depth + 1 >= MAX_DEPTH or random() < 0.3:
                possible_terminals = boolean_terms
                self.left = GPTree(possible_terminals[randint(0, len(possible_terminals) - 1)])
            else:
                possible_functions = boolean_funcs + number_to_boolean_funcs
                self.left = GPTree(possible_functions[randint(0, len(possible_functions) - 1)])
                self.left.tree_constructor(depth + 1)
        elif self.left_type == 'n':
            if not depth + 1 >= MAX_DEPTH or random() < 0.3:
                possible_terminals = random_terms
                self.left = GPTree(possible_terminals[randint(0, len(possible_terminals) - 1)])
            else:
                possible_functions = arithmetic_funcs + price_volume_funcs + price_volume_funcs
                self.left = GPTree(possible_functions[randint(0, len(possible_functions) - 1)])
                self.left.tree_constructor(depth + 1)
        elif self.left_type == 'nn':
            possible_terminals = n_terms
            self.left = GPTree(possible_terminals[randint(0, len(possible_terminals) - 1)])
        else:
            print("in tree constructor, left life is None ! ")


def init_population():
    pop = []
    for i in range(POP_SIZE):
        if random() > 0.5:
            root = GPTree(boolean_funcs[randint(0, len(boolean_funcs) - 1)])
            root.tree_constructor(depth=1)
        else:
            root = GPTree(number_to_boolean_funcs[randint(0, len(number_to_boolean_funcs) - 1)])
            root.tree_constructor(depth=1)
        pop.append(root)
    return pop


def fitness(individual, dataset):  # inverse mean absolute error over dataset normalized to [0,1]
    # print("newfitness")
    print(individual.print_tree())
    value = 0
    prev_buy_not_sell_signal = None
    state = 0
    for time in range(max(n_terms),max(n_terms)+450):
        func.set_time(time)
        buy_not_sell_signal = individual.compute_tree()
        if prev_buy_not_sell_signal is not None:
            if buy_not_sell_signal == prev_buy_not_sell_signal:
                continue
        if buy_not_sell_signal:
            # buying
            value -= price_pseries[time]
            state += 1
        else:
            # selling
            value += price_pseries[time]
            state -=1
        prev_buy_not_sell_signal = buy_not_sell_signal
    if state == 1:
        value += price_pseries[time]
    elif state == -1:
        value -= price_pseries[time]
    return value


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
    time = randint(0,len(price_pseries)-1)
    fitnesses = [fitness(population[i], dataset) for i in range(POP_SIZE)]
    print(fitnesses)

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

