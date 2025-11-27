"""
Basic rules about GOL:
1. Any live cell with one or zero neighbour cells alive then the cell dies, caused by loneliness.
2. if a cell is surrounded by two or three neighbours then it survives
3. Any live cell with more than three live neighbours dies, as if by overpopulation.
4. Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction
"""

import numpy as np
import time
import pygame

colorBg = (10, 10, 10)
# bg will be dark grey
colorGrid = (40, 40, 40)
# grid is also darker, but lighter than bg
colorDieNext = (0, 0, 255)
# white if we are alive and if we're going to stay alove and if we're going to die then we want a lighter grey
colorAliveNext = (191, 64, 191)
# white if we are alive and if we're going to stay alive


# screen is the pygame window
# cells is the 2D array representing the board, state of the individual cells
# size of the individual cell
# with_progress: whether to show progress or not, a boolean
# with_progress=True, means we will apply the game rules and it will move to the next step
def update(screen, cells, size, with_progress=False):
    # purpose: we just want to update the screen with the new state of the cells
    # without moving to the next generate before we calculate the new state
    updated_cells = np.zeros((cells.shape[0], cells.shape[1]))
    # create a new array of zeros with the same shape as cells
    # takes the shape of already existing cells and creates an empty array and based on that we will apply the changes
    for row, col in np.ndindex(cells.shape):
        alive = cells[row, col] == 1
        # neighbors = cells[row - 1 : row + 2, col - 1 : col + 2].sum() - cells[row, col]
        neighbors = (
            np.sum(cells[row - 1 : row + 2, col - 1 : col + 2]) - cells[row, col]
        )
        # calculate the number of the alive neighbouring cells
        # sum of all the neighbours that are alive
        # slicing the 2D array to get the 3x3 block around the cell at (row, col)
        color = colorBg if cells[row, col] == 0 else colorAliveNext

        if alive:
            if neighbors < 2 or neighbors > 3:  # either loneliness or over population
                if with_progress:
                    color = colorDieNext
            elif 2 <= neighbors <= 3:  # the cell stays alive
                updated_cells[row, col] = 1
                if with_progress:
                    color = colorAliveNext
        else:  # that is cell isn't alive but we know that there is a condition of exactly 3 neighbors then it reproduces
            if neighbors == 3:  # reproduction
                updated_cells[row, col] = 1
                if with_progress:
                    color = colorAliveNext

        pygame.draw.rect(
            screen,
            color,
            (col * size, row * size, size - 1, size - 1),  # position of the cell
        )  # draw the cell on the screen
    return updated_cells


def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))

    cells = np.zeros((60, 80))
    # cells = np.zeros((height // 10, width // 10), dtype=int)
    # initial state: all cells are dead
    screen.fill(colorGrid)
    # grid color as the bg, we go to each cell and we fill it to the bg color, since it is the initial state
    update(screen, cells, 10)

    pygame.display.flip()
    # flip the display to update the screen with the new state of the cells
    pygame.display.update()
    # update the display to show the changes made to the screen
    running = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    running = not running
                    update(screen, cells, 10)
                    pygame.display.update()
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                cells[pos[1] // 10, pos[0] // 10] = 1
                update(screen, cells, 10)
                pygame.display.update()
            # above is for a mouse click to toggle cell state by the user
        screen.fill(colorGrid)
        if running:
            cells = update(screen, cells, 10, with_progress=True)
            pygame.display.update()
        time.sleep(0.1)


if __name__ == "__main__":
    main()
