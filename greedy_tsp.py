import random
import matplotlib.pyplot as plt
import math
import numpy as np
import itertools


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        return np.hypot(self.x - city.x, self.y - city.y)

    def __repr__(self):
        return f"({self.x}, {self.y})"


def path_cost(route):
    distance = 0
    for index, city in enumerate(route):
        distance += city.distance(route[(index + 1) % len(route)])
    return distance


class Greedy:
    def __init__(self, cities):
        self.unvisited = cities[1:]
        self.route = [cities[0]]

    def run(self):
        while len(self.unvisited):
            index, nearest_city = min(enumerate(self.unvisited),
                                      key=lambda item: item[1].distance(self.route[-1]))
            self.route.append(nearest_city)
            del self.unvisited[index]
        return path_cost(self.route)


if __name__ == "__main__":
    cities = []
    with open('cities_1024.data', 'r') as handle:
        lines = handle.readlines()
        for line in lines:
            x, y = map(int, line.split())
            cities.append(City(x, y))
    greedy = Greedy(cities)
    # greedy = Greedy([City(x=int(random.random() * 200), y=int(random.random() * 200)) for _ in range(32)])
    print(greedy.run())
    print(greedy.route)
    x = []
    y = []
    fig = plt.figure(0)
    fig.suptitle('Greedy TSP')
    x_list, y_list = [], []
    for city in greedy.route:
        x_list.append(city.x)
        y_list.append(city.y)
    x_list.append(greedy.route[0].x)
    y_list.append(greedy.route[0].y)

    plt.plot(x_list, y_list, 'ro')
    plt.plot(x_list, y_list)
    plt.show(block=True)

    plt.plot(x, y, 'ro')
    plt.show()
    # divideConquer.shows_particles()
    # print(f'gbest: {pso.gbest.pbest}\t| cost: {pso.gbest.pbest_cost}')
    # plt.plot(pso.gcost_iter)
    # plt.ylabel('Distance')
    # plt.xlabel('Generation')
    # plt.show()
