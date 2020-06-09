from configs import NUM_ROADS, NUM_CARS, LAMBDA_VAL

def construct_tsp_matrix(dist_matrix):
    size = len(dist_matrix)
    return add_uniqueness_constraint(size,
        add_existence_constraint(size, size,
            add_costs_tsp(dist_matrix, 
                initialize_diagonal_elements(size, size)
            )
        )
    )

def construct_traffic_matrix(share_pairs):
    return add_existence_constraint(NUM_ROADS, NUM_CARS,
        add_costs_traffic(share_pairs, NUM_ROADS, 
            initialize_diagonal_elements(NUM_ROADS, NUM_CARS)
        )
    )

def initialize_diagonal_elements(group_size, number_of_groups):
    Q = {}
    for current_group in range(group_size):
        for current_city in range(number_of_groups):
            offset = current_group * group_size
            q = offset + current_city
            Q[(q, q)] = 0
    return Q

def add_costs_tsp(dist_matrix, matrix):
    Q = matrix
    size = len(dist_matrix)
    for current_group in range(size):
        for current_city in range(size):
            for next_city in range(size):
                if current_city == next_city:
                    continue
                offset1 = current_group * size
                offset2 = ((current_group + 1) % size) * size
                q1 = offset1 + current_city
                q2 = offset2 + next_city
                Q[(q1, q2)] = dist_matrix[current_city][next_city]
    return Q

def add_costs_traffic(share_pairs, num_roads, matrix):
    Q = matrix
    for share_pair in share_pairs:
        car1_data = share_pair[0]
        car1 = car1_data[0]
        road1 = car1_data[1]
        car2_data = share_pair[1]
        car2 = car2_data[0]
        road2 = car2_data[1]
        q1 = (car1 - 1) * num_roads + (road1 - 1)
        q2 = (car2 - 1) * num_roads + (road2 - 1)
        Q[(q1, q1)] += 1
        Q[(q2, q2)] += 1
        if (q1, q2) not in Q.keys():
            Q[(q1, q2)] = 2
        else:
            Q[(q1, q2)] += 2
    return Q

def add_existence_constraint(group_size, number_of_groups, matrix):
    Q = matrix
    for current_group in range(group_size):
        for current_city in range(number_of_groups):
            group_offset = current_group * group_size
            q1 = group_offset + current_city
            Q[(q1, q1)] += -LAMBDA_VAL
            for other_city in range(number_of_groups):
                q2 = group_offset + other_city
                if current_city != other_city:
                    Q[(q1, q2)] = 2 * LAMBDA_VAL
    return Q

def add_uniqueness_constraint(size, matrix):
    Q = matrix
    for current_city in range(size):
        for current_group in range(size):
            offset1 = current_group * size
            q1 = offset1 + current_city
            Q[(q1, q1)] += -LAMBDA_VAL
            for other_group in range(size):
                offset2 = other_group * size 
                q2 = offset2 + current_city
                if current_group != other_group:
                    Q[(q1, q2)] = 2 * LAMBDA_VAL
    return Q
