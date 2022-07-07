import numpy as np
import os
import argparse
from scipy.optimize import linear_sum_assignment

def test():
    # example cost matrix
    cost0 = np.array([
        [4, 1, 3],
        [2, 0, 5],
        [3, 2, 2]
    ])
    cost = np.array([
        [48,  85,  42,  73,  30],
        [46,  32,  62,  84,  32],
        [79,  50,  86,   9,  84],
        [4,  96,   3,  67,  49],
        [98,  55,  49,  48,  18]
    ])

    # minimize
    row_ind, col_ind = linear_sum_assignment(cost)
    print(col_ind) #array([1, 0, 2])
    print(cost[row_ind, col_ind].sum()) # 5

    # maximize
    row_ind, col_ind = linear_sum_assignment(cost, True)
    print(col_ind) #array([0 2 1])
    print(cost[row_ind, col_ind].sum()) # 11

def hungarian(cost, mode):

    if mode == 1 : # maximize
        row_ind, col_ind = linear_sum_assignment(cost, True)
    else: # minimize
        row_ind, col_ind = linear_sum_assignment(cost)

    return col_ind, cost[row_ind, col_ind].sum()

def compare(output_py, output_c):
    for idx in range(len(output_py)):
        if output_py[idx] != output_c[idx]:
            print("[ERROR] Something wrong ... data is not matched!!!")
            return False
    print("[INFO] All data is same!")


if __name__ == '__main__':

    if 1 :
        parser = argparse.ArgumentParser(description='parameters')
        parser.add_argument('--H', type=int, default=5, help='height')
        parser.add_argument('--W', type=int, default=10,help='width')
        #parser.add_argument('--M', type=bool, default=True, help='maximize')
        args = parser.parse_args()

        dir_path = os.path.dirname(__file__)
        H = args.H
        W = args.W
        #mode = args.M

        input_matrix = np.fromfile(os.path.join(dir_path, 'costMatrix'), dtype=np.float32)
        input_matrix = input_matrix.reshape(H, W)

        # minimize
        output_0_py, tot0 = hungarian(input_matrix, 0)
        output_0_c = np.fromfile(os.path.join(dir_path, 'index_0'), dtype=np.float32).astype(np.int32)

        # maximize
        output_1_py, tot1  = hungarian(input_matrix, 1)
        output_1_c = np.fromfile(os.path.join(dir_path, 'index_1'), dtype=np.float32).astype(np.int32)

        # value compare
        print('[INFO] mode 0 인 경우')
        compare(output_0_py, output_0_c)
        print('[INFO] python : {}, {}'.format(output_0_py, tot0))
        print('[INFO] c++    : {}'.format(output_0_c))

        print('[INFO] mode 1 인 경우')
        compare(output_1_py, output_1_c)
        print('[INFO] python : {}, {}'.format(output_1_py, tot1))
        print('[INFO] c++    : {}'.format(output_1_c))


    else:
        test()