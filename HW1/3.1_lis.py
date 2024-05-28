from typing import Tuple, List
from argparse import ArgumentParser


def longest_increasing_subsequence(arr: List[int]) -> Tuple[int, List[int]]:
    """
    Finds the length and elements of the longest increasing subsequence in a given list.
    It uses dynamic programming to solve the problem in O(n^2) time complexity.
    Args:
        arr (List[int]): The input list of integers.
    Returns:
        Tuple[int, List[int]]: A tuple containing the length of the longest increasing subsequence
        and the elements of the subsequence in the order they appear in the original list.
    """
    n = len(arr)
    # dp[i] stores the length of the longest increasing subsequence ending at index i
    dp = [1] * n
    # prev[i] stores the index of the element before arr[i] in the longest increasing subsequence
    prev = [-1] * n  # init to -1 to indicate the start of the subsequence

    # iterate over the list to find the longest increasing subsequence by
    for i in range(1, n):
        for j in range(i):
            if arr[j] < arr[i]:
                if dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    prev[i] = j

    max_len = max(dp)
    max_idx = dp.index(max_len)

    lis = []
    while max_idx != -1:
        # insert at the beginning of the list to maintain the order
        lis.insert(0, arr[max_idx])
        # move to the previous element in the subsequence
        max_idx = prev[max_idx]

    return max_len, lis


def _get_parameters():
    parser = ArgumentParser()

    default_list = [3, 1, 5, 3, 4]

    parser.add_argument('-i', '--input', type=int, nargs='+', default=default_list,
                        help='The list of integers to find the longest increasing subsequence of')
    return parser.parse_args()


if __name__ == "__main__":

    args = _get_parameters()
    in_list = args.input

    max_len, lis = longest_increasing_subsequence(in_list)
    print(f"The longest increasing subsequence is {lis} with length {max_len}")
