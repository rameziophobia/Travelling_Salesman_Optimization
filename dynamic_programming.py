import itertools
import math

import matplotlib.pyplot as plt
from util import City, read_cities, write_cities_and_return_them, generate_cities, path_cost


def path_cost(route):
    distance = 0
    for index, city in enumerate(route):
        distance += city.distance(route[(index + 1) % len(route)])
    return distance


class DynamicProgramming:
    def __init__(self, cities):
        self.distance_matrix = self.generate_distance_matrix(cities)

    def run(self):
        n = len(self.distance_matrix)
        all_points_set = set(range(n))

        # memo keys: tuple(sorted_points_in_path, last_point_in_path)
        # memo values: tuple(cost_thus_far, next_to_last_point_in_path)
        memo = {(tuple([i]), i): tuple([0, None]) for i in range(n)}
        queue = [(tuple([i]), i) for i in range(n)]

        while queue:
            prev_visited, prev_last_point = queue.pop(0)
            prev_dist, _ = memo[(prev_visited, prev_last_point)]
            to_visit = all_points_set.difference(set(prev_visited))

            for new_last_point in to_visit:
                new_visited = tuple(sorted(list(prev_visited) + [new_last_point]))
                new_dist = (prev_dist + self.distance_matrix[prev_last_point][new_last_point])

                if (new_visited, new_last_point) not in memo:
                    memo[(new_visited, new_last_point)] = (new_dist, prev_last_point)
                    queue += [(new_visited, new_last_point)]
                else:
                    if new_dist < memo[(new_visited, new_last_point)][0]:
                        memo[(new_visited, new_last_point)] = (new_dist, prev_last_point)

        optimal_path, optimal_cost = self.retrace_optimal_path(memo, n)
        return optimal_path, optimal_cost

    def retrace_optimal_path(self, memo, n):
        points_to_retrace = tuple(range(n))
        full_path_memo = dict((k, v) for k, v in memo.items()
                              if k[0] == points_to_retrace)
        path_key = min(full_path_memo.keys(), key=lambda x: full_path_memo[x][0])

        last_point = path_key[1]
        optimal_cost, next_to_last_point = memo[path_key]
        optimal_path = [last_point]

        points_to_retrace = tuple(sorted(set(points_to_retrace).difference({last_point})))
        while next_to_last_point is not None:
            last_point = next_to_last_point
            path_key = (points_to_retrace, last_point)
            _, next_to_last_point = memo[path_key]

        optimal_path = [last_point] + optimal_path
        optimal_cost += sorted([dist for dist in self.distance_matrix[last_point] if dist != 0])[2]
        return optimal_path, optimal_cost

    def plot(self):
        fig = plt.figure(0)
        fig.suptitle('Dynamic Programming TSP')
        x_list, y_list = [], []
        for city in [*self.route, *self.unvisited]:
            x_list.append(city.x)
            y_list.append(city.y)
        x_list.append(self.route[0].x)
        y_list.append(self.route[0].y)

        plt.plot(x_list, y_list, 'ro')
        plt.show(block=False)

    @staticmethod
    def generate_distance_matrix(cities):
        distance_matrix = [[] for _ in range(len(cities))]
        for index, city in enumerate(cities):
            for city_2 in cities:
                distance_matrix[index].append(math.hypot(city_2.x - city.x, city_2.y - city.y))
        return distance_matrix


if __name__ == "__main__":

    cities = write_cities_and_return_them(22)


    def solve_tsp_dynamic(points):
        # calc all lengths
        all_distances = [[x.distance(y) for y in points] for x in points]
        # initial value - just distance from 0 to every other point + keep the track of edges
        A = {(set([0, idx + 1]), idx + 1): (dist, [0, idx + 1]) for idx, dist in enumerate(all_distances[0][1:])}
        cnt = len(points)
        for m in range(2, cnt):
            B = {}
            for S in [set(C) | {0} for C in itertools.combinations(range(1, cnt), m)]:
                for j in S - {0}:
                    B[(S, j)] = min([(A[(S - {j}, k)][0] + all_distances[k][j], A[(S - {j}, k)][1] + [j]) for k in S if
                                     k != 0 and k != j])  # this will use 0th index of tuple for ordering, the same as if key=itemgetter(0) used
            A = B
        res = min([(A[d][0] + all_distances[0][d[1]], A[d][1]) for d in iter(A)])
        return res[1]


    g = solve_tsp_dynamic(cities)
    sol = [cities[gi] for gi in g]
    print(path_cost(sol))
    # dynamic = DynamicProgramming(cities)
    #
    # print(dynamic.run())
    plt.show(block=True)
