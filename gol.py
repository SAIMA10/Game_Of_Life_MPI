#!/usr/bin/env python
# mpirun -np 4 python -m mpi4py life_from_ring.py --nx 200 --ny 120 --steps 500 --print-every 50
import argparse
import numpy as np
from mpi4py import MPI
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_frame_png(path, arr):
    plt.figure(figsize=(6, 6))
    plt.imshow(arr, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=16384, help="columns (X)")
    p.add_argument("--ny", type=int, default=16384, help="rows (Y)")
    p.add_argument("--steps", type=int, default=2000, help="generations")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print-every", type=int, default=0)
    return p.parse_args()


def neighbors_1d_ring(rank, size):
    up = (rank - 1) % size
    down = (rank + 1) % size
    return up, down


# this will calculate the local rows - how many rows this specific rank gets
# row0 which global row this rank's part begins at
def decompose_rows(ny, size, rank):
    base = ny // size  # number of rows each process gets if they were evenly divided.
    rem = ny % size  # remaining rows
    local_rows = base + (1 if rank < rem else 0)
    # local rows for that rank, if there are rem, then it'll take that too
    row0 = rank * base + min(rank, rem)
    # the starting row index for the rank
    return row0, local_rows


# seed Gosper gun
def seed_gosper(local, nx, row0, local_rows):
    gun = np.array(
        [
            [0, 24],
            [1, 22],
            [1, 24],
            [2, 12],
            [2, 13],
            [2, 20],
            [2, 21],
            [2, 34],
            [2, 35],
            [3, 11],
            [3, 15],
            [3, 20],
            [3, 21],
            [3, 34],
            [3, 35],
            [4, 0],
            [4, 1],
            [4, 10],
            [4, 16],
            [4, 20],
            [4, 21],
            [5, 0],
            [5, 1],
            [5, 10],
            [5, 14],
            [5, 16],
            [5, 17],
            [5, 22],
            [5, 24],
            [6, 10],
            [6, 16],
            [6, 24],
            [7, 11],
            [7, 15],
            [8, 12],
            [8, 13],
        ],
        dtype=int,
    )
    pattern_row_start, pattern_column_start = (
        150,
        150,
    )  # define where the pattern should start

    # sets the global coordinates of the pattern,
    # the offets given above added to the top left corner (initial state given)
    global_rows = pattern_row_start + gun[:, 0]
    # periodic boundary conditions, from the right edge to the left edge
    global_cols = (pattern_column_start + gun[:, 1]) % nx

    in_local_chunk = (global_rows >= row0) & (global_rows < row0 + local_rows)
    # calculates the cells which lie within the local grid, by checking the global rows
    # of the pattern with the row0 (where the local grid starts) and where it ends (row0 + local_rows
    # in_local_chunk is a boolean mask

    if not np.any(in_local_chunk):
        return

    rr = (global_rows[in_local_chunk] - row0) + 1  # +1 to skip top halo
    # converting the global_rows indices -> local_row indices
    # Subtract row0 to make them relative to the start of this slab.
    # Add +1 because local row 0 is reserved for the top halo (the first real row is local index 1).

    cc = global_cols[in_local_chunk]
    # no change as its a row decomposition
    local[rr, cc] = 1
    # mark the cells as live


# life_step computes next gen only for interior rows (1..-2) and writes into nxt.
def life_step(curr, nxt):
    # curr/nxt shapes: (local_rows+2, nx); halos in first/last rows
    # curr → the current generation of cells (each either 0=dead or 1=alive).
    # nxt → an empty array where it will write the next generation.

    # shifts elements along an axis, wrapping around at the ends, axis=1 means move things left or right
    left = np.roll(curr, 1, axis=1)  # shift right
    right = np.roll(curr, -1, axis=1)  # shift left
    # we need to know the left/right neighbours of the cells, so we shift the entire array to get all neighbours at once
    # in numpy, we shift right/left
    # np.roll conveniently handles horizontal wrapping without needing extra MPI communication.

    U = curr[0:-2, :]
    # rows just above each cell’s row (the “upper” neighbors)
    C = curr[1:-1, :]
    # the interior rows we will actually update, current rows (the ones we update)
    D = curr[2:, :]
    # rows below interior, rows just below (the “lower” neighbors)

    nbrs = (
        left[0:-2, :]
        + U
        + right[0:-2, :]
        + left[1:-1, :]
        + right[1:-1, :]
        + left[2:, :]
        + D
        + right[2:, :]
    )
    # upper-left, upper, upper-right
    # left and right neighbors
    # lower-left, lower, lower-right
    # here we are calculating all the 8 cell neighbours
    nxt[1:-1, :] = ((nbrs == 3) | ((C == 1) & (nbrs == 2))).astype(np.uint8)
    # applying the game rules
    # nxt[1:-1, :] only update interior rows, not the halo rows (those will be overwritten by halo exchange in the next step).


# we need to gather the data from the MPI ranks and put it into rank0,
# why we do that is because we need to plot it and matplotlib isn't distributed
# so we need to know the entire grid data to plot it.
# in Gatherv, how many rows each rank will send → counts AND
# where in the final array each rank’s data starts → displs (displacements)
def counts_displs(ny, size):
    base, rem = ny // size, ny % size
    counts = [base + (1 if r < rem else 0) for r in range(size)]

    displs = [sum(counts[:r]) for r in range(size)]
    return np.array(counts), np.array(displs)


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    nx, ny = args.nx, args.ny

    # Decompose along rows
    row0, local_rows = decompose_rows(ny, size, rank)
    counts, displs = counts_displs(ny, size)

    full = np.empty((ny, nx), dtype=np.uint8) if rank == 0 else None

    # Allocate with halo rows (top/bottom).
    # setting up the memory for the MPI process
    curr = np.zeros((local_rows + 2, nx), dtype=np.uint8)
    # its current grid slice (its part of the world for this step)
    nxt = np.zeros_like(curr)
    # an empty array of the same shape that will store the next generation

    # Seed pattern by global coordinates; only owning ranks place cells, ensures that any future random actions will be reproducible
    np.random.seed(args.seed)
    seed_gosper(curr, nx, row0, local_rows)

    # everyone is ready before timing
    comm.Barrier()
    t0 = MPI.Wtime()  # start timer (seconds since some MPI-defined time)

    # neighbors, up and down
    up, down = neighbors_1d_ring(rank, size)

    # --- 4) Time stepping ---
    for step in range(args.steps):
        # Halo exchange using your exact pattern: Irecv/Irecv + Isend/Isend + Waitall
        reqs = []
        # reqs is just a list of MPI request objects that represent the non-blocking communication operations

        # Receive from up/down into our halo rows
        reqs.append(comm.Irecv(curr[0, :], source=up, tag=1))
        # top halo from 'up', top halo row (not our data). We receive our UP neighbor’s last real row into here
        reqs.append(comm.Irecv(curr[-1, :], source=down, tag=0))
        # bottom halo from 'down', bottom halo row (not our data). We receive our DOWN neighbor’s first real row into here.

        # Send our interior border rows upward/downward
        reqs.append(comm.Isend(curr[1, :], dest=up, tag=0))
        # send first interior row up, our first real (interior) row. You send this up so your UP neighbor can use it as their bottom halo.
        reqs.append(comm.Isend(curr[-2, :], dest=down, tag=1))
        # send last interior row down, our last real (interior) row. We send this down so our DOWN neighbor can use it as their top halo.

        MPI.Request.Waitall(reqs)
        # ensures halos are in place before compute.

        # Compute next generation on interior
        life_step(curr, nxt)
        # After compute, you swap so curr always holds the current generation and
        # nxt becomes the scratch buffer for the next loop.

        # swap buffers
        curr, nxt = nxt, curr

        sendbuf = curr[1:-1, :]
        # remove the halo rows (the one on top and bottom) and keep the interior rows
        comm.Gatherv(sendbuf, (full, counts * nx, displs * nx, MPI.BYTE), root=0)
        # if rank == 0:
        #     plt.imshow(full, cmap="gray", interpolation="nearest")
        #     plt.axis("off")
        #     plt.tight_layout(pad=0)
        #     plt.savefig(f"frame_root0.png", dpi=150, bbox_inches="tight", pad_inches=0)
        #     plt.close()

        if rank == 0 and args.print_every and (step % args.print_every == 0):
            save_frame_png(f"frame_1D_{step:05d}.png", full)

        # population checksum to stdout
        if args.print_every and (step % args.print_every == 0):
            local_pop = int(curr[1:-1, :].sum())
            # the population of the cells (alive) on the local grid
            total_pop = comm.allreduce(local_pop, op=MPI.SUM)
            # MPI collective operation: adds up (SUM) all ranks’ local_pop values.
            if rank == 0:
                print(f"step {step:6d} population={total_pop}")

    comm.Barrier()  # ensure all ranks finished the loop
    t1 = MPI.Wtime()

    loop_time = t1 - t0
    # Final checksum
    local_pop = int(curr[1:-1, :].sum())
    total_pop = comm.allreduce(local_pop, op=MPI.SUM)
    if rank == 0:
        print(f"Done. final population={total_pop}")
        print(
            f"RANKS={size} STEPS={args.steps} TIME_LOOP={loop_time:.6f} s "
            f"TIME_PER_STEP={loop_time/args.steps:.6e} s"
        )


if __name__ == "__main__":
    main()
