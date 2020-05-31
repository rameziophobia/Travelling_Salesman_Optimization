import math

import matplotlib.pyplot as plt
import numpy as np
import operator
import pandas as pd
import random


class City:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        return np.hypot(abs(self.x - city.x), abs(self.y - city.y))

    def __repr__(self):
        return f"({self.x}, {self.y})"


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def path_cost(self):
        if self.distance == 0:
            distance = 0
            for index, city in enumerate(self.route):
                distance += city.distance(self.route[(index + 1) % len(self.route)])
            self.distance = distance
        return self.distance

    def path_fitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.path_cost())
        return self.fitness


def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route


def initial_population(popSize, cityList):
    random_population = [createRoute(cityList) for _ in range(popSize)]
    # greedy_population = [greedy_route(start_index % len(cityList), cityList)
    #                      for start_index in range(math.ceil(popSize // 10))]
    return [*random_population]


def greedy_route(start_index, cities):
    unvisited = cities[:]
    del unvisited[start_index]
    route = [cities[start_index]]
    while len(unvisited):
        index, nearest_city = min(enumerate(unvisited), key=lambda item: item[1].distance(route[-1]))
        route.append(nearest_city)
        del unvisited[index]
    return route


def rank_chromosomes(population):
    fitness = [(i, Fitness(population[i]).path_fitness()) for i in range(len(population))]
    return sorted(fitness, key=lambda f: f[1], reverse=True)


def selection(ranked_population, num_elites):
    selectionResults = []
    df = pd.DataFrame(np.array(ranked_population), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, num_elites):
        selectionResults.append(ranked_population[i][0])
    for i in range(0, len(ranked_population) - num_elites):
        pick = 100 * random.random()
        for i in range(0, len(ranked_population)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(ranked_population[i][0])
                break
    return selectionResults


def mating_pool(population, selection_results):
    matingpool = []
    for i in range(0, len(selection_results)):
        index = selection_results[i]
        matingpool.append(population[index])
    return matingpool


def breed(parent1, parent2):
    gene_1 = random.randint(0, len(parent1))
    gene_2 = random.randint(0, len(parent1))
    gene_1 = min(gene_1, gene_2)
    gene_2 = max(gene_1, gene_2)
    child = [parent1[i] for i in range(gene_1, gene_2)]
    child.extend([gene for gene in parent2 if gene not in child])
    return child


def breed_population(mating_pool, num_elites):
    children = mating_pool[:num_elites]
    for i in range(0, len(mating_pool) - num_elites):
        child = breed(mating_pool[i], mating_pool[(i + random.randint(1, num_elites)) % (len(mating_pool) - num_elites)])
        children.append(child)
    return children


def mutate(individual, mutation_rate):
    for index, city in enumerate(individual):
        if random.random() < mutation_rate:
            random_index = random.randint(0, len(individual) - 1)
            individual[index], individual[random_index] = individual[random_index], individual[index]
    return individual


def next_generation(this_generation, elites_num, mutation_rate):
    pop_ranked = rank_chromosomes(this_generation)
    selection_results = selection(pop_ranked, elites_num)
    matingpool = mating_pool(this_generation, selection_results)
    children = breed_population(matingpool, elites_num)
    nextGeneration = [mutate(chromosome, mutation_rate) for chromosome in children]
    return nextGeneration


cities = [City(x=int(random.random() * 200), y=int(random.random() * 200)) for _ in range(64)]
cities = []
with open('cities_64.data', 'r') as handle:
    lines = handle.readlines()
    for line in lines:
        x, y = map(int, line.split())
        cities.append(City(x, y))


def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initial_population(popSize, population)
    progress = []
    progress.append(1 / rank_chromosomes(pop)[0][1])

    for i in range(0, generations):
        pop = next_generation(pop, eliteSize, mutationRate)
        progress.append(1 / rank_chromosomes(pop)[0][1])

    print("Final distance: " + str(1 / rank_chromosomes(pop)[0][1]))
    bestRouteIndex = rank_chromosomes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]

    plt.figure(0)
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')

    x_list, y_list = [], []
    for city in bestRoute:
        x_list.append(city.x)
        y_list.append(city.y)
    x_list.append(bestRoute[0].x)
    y_list.append(bestRoute[0].y)
    fig = plt.figure(1)
    fig.suptitle('genetic algorithm TSP')
    plt.plot(x_list, y_list, 'ro')
    plt.plot(x_list, y_list)

    plt.show()
    return bestRoute


geneticAlgorithmPlot(population=cities, popSize=110, eliteSize=20, mutationRate=0.005, generations=1000)
