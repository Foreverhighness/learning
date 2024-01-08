#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import os
import random
from typing import List, Tuple


NUMBERS = 1024
BUILD_DIR = 'build'


def parse_cmd() -> Tuple[int, bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument('num_file', type=int, nargs='?', default=1000)
    parser.add_argument('-r', '--random', action='store_true')
    args = parser.parse_args()

    num_file = args.num_file
    init_with_random = args.random
    return num_file, init_with_random


def prepare_data(num_file: int, init_with_random: bool) -> List[int]:
    data = []
    count = num_file * NUMBERS
    if init_with_random:
        data = [random.randint(0, 2**32 - 1) for _ in range(count)]
    else:
        data = random.sample(range(1, count + 1), count)
    return data


def output_data(data: List[int], num_file: int):
    if not os.path.exists(BUILD_DIR):
        os.makedirs(BUILD_DIR)

    i = 0
    for filename in range(num_file):
        with open(f'{BUILD_DIR}/{filename}.bin', 'wb') as bin_file:
            for _ in range(NUMBERS):
                bin_file.write(data[i].to_bytes(4, 'little'))
                i += 1


def main():
    num_file, init_with_random = parse_cmd()
    data = prepare_data(num_file, init_with_random)
    output_data(data, num_file)


if __name__ == '__main__':
    main()
