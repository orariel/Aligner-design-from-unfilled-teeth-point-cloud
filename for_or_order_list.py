import copy
from math import sqrt
import numpy as np


# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(1, len(row1)):  # Ils mettent -1, car ils ne prennent pas en compte le z.
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])  # sort selon l'element dist parmi les 2 elements train_row and dist
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])  # on prend train_row et pas dist
    return neighbors


def get_sorted_arr(arr_to_sort,first_index):
    dataset = arr_to_sort
    neighbors = []  # this list aims to collect the neighbors data
    order_list = []  # this list aims to collect the order labels according to distances
    index_first = first_index  # first index (one of side minimum), for exemple index_first =10
    order_list.append(index_first)
    dataset_removed = copy.deepcopy(
        dataset)  # this dataset will be the same of your original one but you will replace values during the process
    dataset_removed[order_list[0]] = [1000, 1000, 1000]
    neighbors = get_neighbors(dataset_removed, dataset[order_list[0]], 1)

    # To find the index of the dataset for the values of the neighbors
    cond1 = np.logical_and(dataset[:, 0] == neighbors[0][0], dataset[:, 1] == neighbors[0][1])
    cond2 = np.logical_and(cond1, dataset[:, 2] == neighbors[0][2])
    index_pos = np.where(cond2)

    # add this index in the list
    order_list.append(index_pos[0][0])
    dataset_removed[order_list[1]] = [1000, 1000, 1000]

    # make the same thing in the following loop for
    for index_range in range(1, len(dataset)):
        print("Loop step : ", index_range)
        dataset_removed[order_list[index_range]] = [1000, 1000, 1000]
        neighbors = get_neighbors(dataset_removed, dataset[order_list[index_range]], 1)

        cond1 = np.logical_and(dataset[:, 0] == neighbors[0][0], dataset[:, 1] == neighbors[0][1])
        cond2 = np.logical_and(cond1, dataset[:, 2] == neighbors[0][2])
        index_pos = np.where(cond2)

        if index_range == len(dataset) - 1:
            break

        print("Index of the neighbor find : ", index_pos[0][0])
        order_list.append(index_pos[0][0])
        print("Values of the neigbor find : ", neighbors[0])



    return order_list
