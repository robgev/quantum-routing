from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from configs import NUM_READS, NUM_ROADS, SAPI_TOKEN, CHAIN_STRENGTH, PROBLEMS

import itertools
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import neal

def calculate_cost_tsp(distances, order):
    cost = 0
    size = len(order)
    for current_idx in range(size):
        next_idx = (current_idx + 1) % len(order) # for closing the loop
        current_city = order[current_idx]
        next_city = order[next_idx]
        cost += distances[current_city][next_city]
    return cost

def calculate_cost_traffic(share_pairs, decisions):
    cost = 0
    decision_roads = list(decisions.values())
    for i in range(len(decision_roads)):
        car1_pair = (i + 1, decision_roads[i])
        for j in range(i + 1, len(decision_roads)):
            car2_pair = (j + 1, decision_roads[j])
            if (car1_pair, car2_pair) in share_pairs or (car2_pair, car1_pair) in share_pairs:
                cost += 1
    return cost

def binary_to_decisions(binary_result, num_roads):
    decisions = {}
    num_cars = int(len(binary_result) / num_roads)
    for car in range(num_cars):
        for road in range(num_roads):
            offset = car * num_roads
            if binary_result[offset + road] == 1:
                decisions[f"car{car + 1}"] = road + 1
    return decisions

def binary_to_order(binary_result):
    order = []
    num_points = int(np.sqrt(len(binary_result)))
    for current_group in range(num_points):
        for current_point in range(num_points):
            offset = current_group * num_points
            if binary_result[offset + current_point] == 1:
                order.append(current_point)
    return order

def solve(Q, problem, is_simulated):
    print('IS_SIMULATED', is_simulated)
    sampler = neal.SimulatedAnnealingSampler() if  is_simulated else EmbeddingComposite(DWaveSampler(token=SAPI_TOKEN, endpoint=ENDPOINT, solver='DW_2000Q_2_1'))
    response = sampler.sample_qubo(Q, chain_strength=CHAIN_STRENGTH, num_reads=NUM_READS)
    return decode_solution(response, problem)
    
def decode_solution(response, problem):
    distribution = {}
    best_solution = []
    # array of tuples (sample array, energy, num_occurences)
    # more info here: 
    # https://docs.ocean.dwavesys.com/en/latest/docs_dimod/reference/sampleset.html#dimod.SampleSet
    records = response.record 
    min_energy = records[0].energy

    for record in records:
        sample = record[0] # extract the sample from the recarray
        binary_result = [node for node in sample] 
        solution = binary_to_order(binary_result) if problem == PROBLEMS['TSP'] else binary_to_decisions(binary_result, NUM_ROADS)
        distribution[tuple(solution)] = (record.energy, record.num_occurrences)
        if record.energy <= min_energy:
            best_solution = solution
    return best_solution, distribution

def plot_solution(name, nodes_array, solution):
    plt.scatter(nodes_array[:, 0], nodes_array[:, 1], s=200)
    for i in range(len(nodes_array)):
        plt.annotate(i, (nodes_array[i, 0] + 0.15, nodes_array[i, 1] + 0.15), size=16, color='r')

    plt.xlim([min(nodes_array[:, 0]) - 1, max(nodes_array[:, 0]) + 1])
    plt.ylim([min(nodes_array[:, 1]) - 1, max(nodes_array[:, 1]) + 1])
    for i in range(len(solution)):
        a = i%len(solution)
        b = (i+1)%len(solution)
        A = solution[a]
        B = solution[b]
        plt.plot([nodes_array[A, 0], nodes_array[B, 0]], [nodes_array[A, 1], nodes_array[B, 1]], c='r')

    cost = calculate_cost(get_tsp_matrix(nodes_array), solution)
    title_string = "Cost:" + str(cost)
    title_string += "\n" + str(solution)
    plt.title(title_string)
    plt.savefig(name + '.png')
    plt.clf()


# print(calculate_cost_traffic([((1,1), (2,1)), ((1,2), (2,1)), ((1,3), (2,1)), ((3, 1), (2,2)), ((3, 2), (1,3)), ((3, 2), (1,1))], {'car1': 1, 'car2': 1, 'car3': 2}))