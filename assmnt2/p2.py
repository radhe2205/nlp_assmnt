from itertools import combinations

import numpy as np

transition_matrix = np.array([[0.2,0.5,0.2,0.1],
                                [0.3,0.4,0.2,0.1],
                                [0.1,0.3,0.4,0.2],
                                [0.1,0.1,0.3,0.5]])

initial_state = np.array([[0.5, 0.3, 0.2, 0.0]])

def prob_dist_after_n_steps(t_mat, init_mat, num_steps):
    prob_dist = init_mat
    for i in range(num_steps-1):
        prob_dist = np.matmul(prob_dist, t_mat)

    return prob_dist / prob_dist.sum()

def find_count_prob(prob_dist, count):
    if count > len(prob_dist[0]) or count < 0:
        print("Invalid Input")
        raise NotImplementedError("Invalid input.")
    combs = combinations(range(len(prob_dist[0])), count)

    total_prob = 0
    for comb in combs:
        prob = 1
        for i in range(len(prob_dist[0])):
            if i in comb:
                prob *= prob_dist[0,i]
            else:
                prob *= (1 - prob_dist[0,i])
        total_prob += prob
    return total_prob

def main():
    print("Probability of exactly 3 lines being busy after 4 steps:")
    prob_dist = prob_dist_after_n_steps(transition_matrix, initial_state, 4)
    print(prob_dist[0,3])

    print("\n\nProbability distribution after 4 steps:" + str(prob_dist[0]))
    print("Most probable #lines being busy after time step 4: " + str(np.argmax(prob_dist[0])))

if __name__ == "__main__":
    main()
