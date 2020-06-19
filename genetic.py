import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import math, sys
import random
from util import City, read_cities, write_cities_and_return_them, generate_cities


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


class GeneticAlgorithm:
    def __init__(self, iterations, population_size, cities, elites_num, mutation_rate,
                 greedy_seed=0, roulette_selection=True, plot_progress=True):
        self.plot_progress = plot_progress
        self.roulette_selection = roulette_selection
        self.progress = []
        self.mutation_rate = mutation_rate
        self.cities = cities
        self.elites_num = elites_num
        self.iterations = iterations
        self.population_size = population_size
        self.greedy_seed = greedy_seed

        self.population = self.initial_population()
        self.average_path_cost = 1
        self.ranked_population = None

    def best_chromosome(self):
        return self.ranked_population[0][0]

    def best_distance(self):
        return 1 / self.ranked_population[0][1]

    def random_route(self):
        return random.sample(self.cities, len(self.cities))

    def initial_population(self):
        p1 = [self.random_route() for _ in range(self.population_size - self.greedy_seed)]
        greedy_population = [greedy_route(start_index % len(self.cities), self.cities)
                             for start_index in range(self.greedy_seed)]
        return [*p1, *greedy_population]

    def rank_population(self):
        fitness = [(chromosome, Fitness(chromosome).path_fitness()) for chromosome in self.population]
        self.ranked_population = sorted(fitness, key=lambda f: f[1], reverse=True)

    def selection(self):
        selections = [self.ranked_population[i][0] for i in range(self.elites_num)]
        if self.roulette_selection:
            df = pd.DataFrame(np.array(self.ranked_population), columns=["index", "fitness"])
            self.average_path_cost = sum(1 / df.fitness) / len(df.fitness)
            df['cum_sum'] = df.fitness.cumsum()
            df['cum_perc'] = 100 * df.cum_sum / df.fitness.sum()

            for _ in range(0, self.population_size - self.elites_num):
                pick = 100 * random.random()
                for i in range(0, len(self.ranked_population)):
                    if pick <= df.iat[i, 3]:
                        selections.append(self.ranked_population[i][0])
                        break
        else:
            for _ in range(0, self.population_size - self.elites_num):
                pick = random.randint(0, self.population_size - 1)
                selections.append(self.ranked_population[pick][0])
        self.population = selections

    @staticmethod
    def produce_child(parent1, parent2):
        gene_1 = random.randint(0, len(parent1))
        gene_2 = random.randint(0, len(parent1))
        gene_1, gene_2 = min(gene_1, gene_2), max(gene_1, gene_2)
        child = [parent1[i] for i in range(gene_1, gene_2)]
        child.extend([gene for gene in parent2 if gene not in child])
        return child

    def generate_population(self):
        length = len(self.population) - self.elites_num
        children = self.population[:self.elites_num]
        for i in range(0, length):
            child = self.produce_child(self.population[i],
                                       self.population[(i + random.randint(1, self.elites_num)) % length])
            children.append(child)
        return children

    def mutate(self, individual):
        for index, city in enumerate(individual):
            if random.random() < max(0, self.mutation_rate):
                sample_size = min(min(max(3, self.population_size // 5), 100), len(individual))
                random_sample = random.sample(range(len(individual)), sample_size)
                sorted_sample = sorted(random_sample,
                                       key=lambda c_i: individual[c_i].distance(individual[index - 1]))
                random_close_index = random.choice(sorted_sample[:max(sample_size // 3, 2)])
                individual[index], individual[random_close_index] = \
                    individual[random_close_index], individual[index]
        return individual

    def next_generation(self):
        self.rank_population()
        self.selection()
        self.population = self.generate_population()
        self.population[self.elites_num:] = [self.mutate(chromosome)
                                             for chromosome in self.population[self.elites_num:]]

    def run(self):
        if self.plot_progress:
            plt.ion()
        for ind in range(0, self.iterations):
            self.next_generation()
            self.progress.append(self.best_distance())
            if self.plot_progress and ind % 10 == 0:
                self.plot()
            elif not self.plot_progress and ind % 10 == 0:
                print(self.best_distance())

    def plot(self):
        print(self.best_distance())
        fig = plt.figure(0)
        plt.plot(self.progress, 'g')
        fig.suptitle('genetic algorithm generations')
        plt.ylabel('Distance')
        plt.xlabel('Generation')

        x_list, y_list = [], []
        for city in self.best_chromosome():
            x_list.append(city.x)
            y_list.append(city.y)
        x_list.append(self.best_chromosome()[0].x)
        y_list.append(self.best_chromosome()[0].y)
        fig = plt.figure(1)
        fig.clear()
        fig.suptitle('genetic algorithm TSP')
        plt.plot(x_list, y_list, 'ro')
        plt.plot(x_list, y_list, 'g')

        if self.plot_progress:
            plt.draw()
            plt.pause(0.05)
        plt.show()


def greedy_route(start_index, cities):
    unvisited = cities[:]
    del unvisited[start_index]
    route = [cities[start_index]]
    while len(unvisited):
        index, nearest_city = min(enumerate(unvisited), key=lambda item: item[1].distance(route[-1]))
        route.append(nearest_city)
        del unvisited[index]
    return route


if __name__ == '__main__':
    cities = read_cities(64)
    genetic_algorithm = GeneticAlgorithm(cities=cities, iterations=1200, population_size=100,
                                         elites_num=20, mutation_rate=0.008, greedy_seed=1,
                                         roulette_selection=True, plot_progress=True)
    genetic_algorithm.run()
    print(genetic_algorithm.best_distance())
    genetic_algorithm.plot()
    plt.show(block=True)
