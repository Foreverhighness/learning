#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys
from typing import List


def parse_args() -> List[str]:
    argv = sys.argv
    files = [argv[i] for i in range(1, len(argv))]
    for path in files:
        if not os.path.exists(path):
            assert False, f'{path} not exists.'
    return files


def main():
    files = parse_args()
    for file in files:
        with open(file, 'rb') as f:
            chunk = f.read(4)
            while chunk:
                print(int.from_bytes(chunk, 'little'))
                chunk = f.read(4)


if __name__ == '__main__':
    main()
