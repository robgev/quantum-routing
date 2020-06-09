from qubo_constructor import construct_tsp_matrix, construct_traffic_matrix
from utils import solve
from configs import PROBLEMS

dist_matrix = [
    [0,5,1,7],
    [5,0,4,2],
    [1,4,0,8],
    [7,2,8,0]
]
share_pairs = [((1,1), (2,1)), ((1,2), (2,1)), ((1,3), (2,1)), ((3, 1), (2,2)), ((3, 2), (1,3)), ((3, 2), (1,1))]

best_solution, distribution = solve(construct_tsp_matrix(dist_matrix), PROBLEMS['TSP'], True)
print(best_solution)