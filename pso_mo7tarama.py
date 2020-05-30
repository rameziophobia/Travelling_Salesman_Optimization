import random
import numpy as np
import matplotlib.pyplot as plt


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        return np.hypot(abs(self.x - city.x), abs(self.y - city.y))

    def __repr__(self):
        return f"({self.x}, {self.y})"


class Particle:
    def __init__(self, route, cost=None):
        self.route = route
        self.pbest = route
        self.current_cost = cost if cost else self.path_cost()
        self.pbest_cost = cost if cost else self.path_cost()

        # velocity of a particle is a sequence of 4-tuple
        # (1, 2, 1, 'beta') means SO(1,2), prabability 1 and compares with "beta"
        self.velocity = []

    def clear_velocity(self):
        self.velocity.clear()

    def update_costs_and_pbest(self):
        self.current_cost = self.path_cost()
        if self.current_cost < self.pbest_cost:
            self.pbest = self.route
            self.pbest_cost = self.current_cost

    def path_cost(self):
        distance = 0
        for index, city in enumerate(self.route):
            distance += city.distance(self.route[(index + 1) % len(self.route)])
        return distance


class PSO:

    def __init__(self, iterations, population_size, beta=1.0, alpha=1.0, cities=None):
        self.cities = cities
        self.gbest = None
        self.gcost_iter = []
        self.iterations = iterations
        self.population_size = population_size
        self.particles = []
        self.beta = beta  # the probability that all swap operators in swap sequence (gbest - x(t-1))
        self.alpha = alpha  # the probability that all swap operators in swap sequence (pbest - x(t-1))

        solutions = self.initial_population()
        self.particles = [Particle(route=solution) for solution in solutions]

    def random_route(self):
        return random.sample(self.cities, len(self.cities))

    def initial_population(self):
        return [self.random_route() for _ in range(self.population_size)]

    def shows_particles(self):

        print('Showing particles...\n')
        for particle in self.particles:
            print(f'pbest: {particle.pbest}\t|\tcost pbest: {particle.pbest_cost}\t|'
                  f'\tcurrent solution: {particle.route}\t|'
                  f'\tcost current solution: {particle.current_cost}')
        print()

    def run(self):
        self.gbest = min(self.particles, key=lambda p: p.pbest_cost)
        print(f"initial cost is {self.gbest.pbest_cost}")
        plt.ion()
        plt.show()
        for t in range(self.iterations):
            self.gbest = min(self.particles, key=lambda p: p.pbest_cost)
            if t % 20 == 0:
                x_list, y_list = [], []
                for city in self.gbest.pbest:
                    x_list.append(city.x)
                    y_list.append(city.y)
                x_list.append(pso.gbest.pbest[0].x)
                y_list.append(pso.gbest.pbest[0].y)
                fig = plt.figure(1)
                fig.clear()
                fig.suptitle(f'pso TSP iter {t}')

                plt.plot(x_list, y_list, 'ro')
                plt.plot(x_list, y_list)
                plt.draw()
                plt.pause(.001)
            self.gcost_iter.append(self.gbest.pbest_cost)
            # average_last_100_iter = sum(self.gcost_iter[-100:]) / len(self.gcost_iter[-100:])
            # if abs(average_last_100_iter - self.gbest.pbest_cost) < 1 and t > self.iterations / 10:
            #     break

            for particle in self.particles:
                particle.clear_velocity()
                temp_velocity = []
                gbest = self.gbest.pbest[:]
                new_route = particle.route[:]

                for i in range(len(self.cities)):
                    if new_route[i] != particle.pbest[i]:
                        swap = (i, particle.pbest.index(new_route[i]), self.alpha)
                        temp_velocity.append(swap)
                        new_route[swap[0]], new_route[swap[1]] = \
                            new_route[swap[1]], new_route[swap[0]]

                for i in range(len(self.cities)):
                    if new_route[i] != gbest[i]:
                        swap = (i, gbest.index(new_route[i]), self.beta)
                        temp_velocity.append(swap)
                        gbest[swap[0]], gbest[swap[1]] = gbest[swap[1]], gbest[swap[0]]

                particle.velocity = temp_velocity

                for swap in temp_velocity:
                    if random.random() <= swap[2]:
                        new_route[swap[0]], new_route[swap[1]] = \
                            new_route[swap[1]], new_route[swap[0]]

                particle.route = new_route
                particle.update_costs_and_pbest()


if __name__ == "__main__":
    # cities = [City(x=int(random.random() * 200), y=int(random.random() * 200)) for _ in range(32)]
    # with open('cities_32.data', 'w+') as handle:
    #     for city in cities:
    #         handle.write(f'{city.x} {city.y}\n')
    cities = []
    with open('cities_256.data', 'r') as handle:
        lines = handle.readlines()
        for line in lines:
            x, y = map(int, line.split())
            cities.append(City(x, y))

    pso = PSO(iterations=1200, population_size=300, alpha=0.9, beta=0.8, cities=cities)
    pso.run()
    pso.shows_particles()
    print(f'cost: {pso.gbest.pbest_cost}\t| gbest: {pso.gbest.pbest}')

    plt.figure(0)
    plt.plot(pso.gcost_iter)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    fig = plt.figure(0)
    fig.suptitle('pso iter')
    x_list, y_list = [], []
    for city in pso.gbest.pbest:
        x_list.append(city.x)
        y_list.append(city.y)
    x_list.append(pso.gbest.pbest[0].x)
    y_list.append(pso.gbest.pbest[0].y)
    fig = plt.figure(1)
    fig.suptitle('pso TSP')

    plt.plot(x_list, y_list, 'ro')
    plt.plot(x_list, y_list)
    plt.show(block=True)

