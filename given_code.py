#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# shebang + encoding comment. Can run the file as an executable and declares UTF-8 source.

import matplotlib.pyplot as plt
import numpy as np

dim = 100  # board size 100x100
earth = np.zeros(
    (dim, dim), dtype=int
)  # initialises the board with 0s, with data type integer

print(earth, "earth initialized")

## init
# earth[40:60, 40:60] = 1
# creates a 20x20 block of "alive" cells in the center --> why?

glider = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 1]]
# glider gun pattern - starts from top left corner --> is it random?

offset_x = 6
offset_y = 10
for x, y in glider:
    earth[x + offset_x, y + offset_y] = 1

newEarth = np.zeros((dim, dim), dtype=int)

life = []

for day in range(100):
    plt.matshow(earth)
    for x in range(1, dim - 1):
        for y in range(1, dim - 1):
            alive = earth[x, y] == 1
            neighborhood = earth[x - 1 : x + 2, y - 1 : y + 2]
            neighbors = neighborhood.sum() - earth[x, y]

            if alive and neighbors < 2:
                newEarth[x, y] = 0
            if alive and (neighbors == 2 or neighbors == 3):
                newEarth[x, y] = 1

            if alive and neighbors > 3:  ## die of overcrowding
                newEarth[x, y] = 0
            if not alive and neighbors == 3:
                newEarth[x, y] = 1

    if np.all(earth == newEarth):
        break
    life.append(earth.sum())
    earth = newEarth

plt.matshow(newEarth)
plt.show()

print("Done")
