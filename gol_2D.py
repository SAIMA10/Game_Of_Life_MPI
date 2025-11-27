#!/usr/bin/env python
"""
With 2D decomposition: P x Q grid of ranks (P rows of ranks, Q columns of ranks).
Each rank owns a sub-block: shape ≈ (local_ny + 2, local_nx + 2) (now both row and column halos).
Exchange halos in four directions: up, down, left, right (and optionally corners).
"""
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


def decompose_2d(ny, nx, p_rows, p_cols, prow, pcol):
    # rows
    base_ny = ny // p_rows
    rem_ny = ny % p_rows
    local_ny = base_ny + (1 if prow < rem_ny else 0)
    row0 = prow * base_ny + min(prow, rem_ny)

    # cols
    base_nx = nx // p_cols
    rem_nx = nx % p_cols
    local_nx = base_nx + (1 if pcol < rem_nx else 0)
    col0 = pcol * base_nx + min(pcol, rem_nx)

    return row0, col0, local_ny, local_nx


def seed_gosper(local_grid, ny, nx, row0, col0, local_ny, local_nx):
    gosper = np.array(
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
    pat_row0, pat_col0 = 5, 5  # global starting corner

    global_rows = pat_row0 + gosper[:, 0]
    global_cols = pat_col0 + gosper[:, 1]

    # wrap on torus
    global_rows %= ny
    global_cols %= nx

    # which pattern cells fall into THIS block?
    in_block = (
        (global_rows >= row0)
        & (global_rows < row0 + local_ny)
        & (global_cols >= col0)
        & (global_cols < col0 + local_nx)
    )
    if not np.any(in_block):
        return

    # map global → local (shift by block origin, +1 for halos)
    local_rows = (global_rows[in_block] - row0) + 1
    local_cols = (global_cols[in_block] - col0) + 1

    local_grid[local_rows, local_cols] = 1


def life_step(curr, nxt):
    # curr/nxt: shape (local_ny+2, local_nx+2) with halos

    center = curr[1:-1, 1:-1]

    nbrs = (
        curr[0:-2, 0:-2]
        + curr[0:-2, 1:-1]
        + curr[0:-2, 2:]  # upper row
        + curr[1:-1, 0:-2]
        + curr[1:-1, 2:]  # middle row
        + curr[2:, 0:-2]
        + curr[2:, 1:-1]
        + curr[2:, 2:]  # lower row
    )

    nxt[1:-1, 1:-1] = ((nbrs == 3) | ((center == 1) & (nbrs == 2))).astype(np.uint8)


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
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # size = comm.Get_size()

    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    # Make a 2D process grid: dims[0] * dims[1] == world_size
    dims = MPI.Compute_dims(world_size, 2)  # e.g. 4→[2,2], 8→[2,4] etc.
    periods = [True, True]  # torus in both directions
    comm2d = comm.Create_cart(dims, periods=periods, reorder=True)

    rank = comm2d.Get_rank()
    coords = comm2d.Get_coords(rank)  # (prow, pcol)
    p_rows, p_cols = dims  # number of ranks in each direction
    prow, pcol = coords

    nx, ny = args.nx, args.ny

    row0, col0, local_ny, local_nx = decompose_2d(ny, nx, p_rows, p_cols, prow, pcol)

    local_info = np.array([row0, col0, local_ny, local_nx], dtype="i")
    all_info = None
    if rank == 0:
        all_info = np.empty((world_size, 4), dtype="i")
    comm2d.Gather(local_info, all_info, root=0)
    if rank == 0:
        # all_info[r] = [row0_r, col0_r, local_ny_r, local_nx_r]
        block_sizes = all_info[:, 2] * all_info[:, 3]  # local_ny * local_nx per rank
        counts_flat = block_sizes.astype("i")
        displs_flat = np.zeros(world_size, dtype="i")
        displs_flat[1:] = np.cumsum(counts_flat[:-1])
    else:
        counts_flat = None
        displs_flat = None

    # allocate with halos on BOTH sides
    curr = np.zeros((local_ny + 2, local_nx + 2), dtype=np.uint8)
    nxt = np.zeros_like(curr)

    # counts, displs = counts_displs(ny, size)

    full = np.empty((ny, nx), dtype=np.uint8) if rank == 0 else None

    # Seed pattern by global coordinates; only owning ranks place cells, ensures that any future random actions will be reproducible
    np.random.seed(args.seed)

    seed_gosper(curr, ny, nx, row0, col0, local_ny, local_nx)

    up, down = comm2d.Shift(direction=0, disp=1)  # row neighbors
    left, right = comm2d.Shift(direction=1, disp=1)  # column neighbors

    # --- 4) Time stepping ---
    for step in range(args.steps):
        # Halo exchange using your exact pattern: Irecv/Irecv + Isend/Isend + Waitall
        reqs = []
        reqs.append(comm2d.Irecv(curr[0, 1:-1], source=up, tag=10))  # top halo
        reqs.append(comm2d.Irecv(curr[-1, 1:-1], source=down, tag=11))  # bottom halo

        reqs.append(comm2d.Isend(curr[1, 1:-1], dest=up, tag=11))  # first interior
        reqs.append(comm2d.Isend(curr[-2, 1:-1], dest=down, tag=10))  # last interior

        # 2) left/right (columns, need copies)
        send_left = curr[1:-1, 1].copy()
        send_right = curr[1:-1, -2].copy()
        recv_left = np.empty_like(send_left)
        recv_right = np.empty_like(send_right)

        reqs.append(comm2d.Irecv(recv_left, source=left, tag=20))
        reqs.append(comm2d.Irecv(recv_right, source=right, tag=21))
        reqs.append(comm2d.Isend(send_left, dest=left, tag=21))
        reqs.append(comm2d.Isend(send_right, dest=right, tag=20))

        MPI.Request.Waitall(reqs)

        # fill halo columns
        curr[1:-1, 0] = recv_left  # left halo
        curr[1:-1, -1] = recv_right  # right halo
        # ensures halos are in place before compute.

        # Compute next generation on interior
        life_step(curr, nxt)
        curr, nxt = nxt, curr

        # ---- gather & save frame on rank 0 every print-every steps ----
        if args.print_every and (step % args.print_every == 0):
            # Flatten our interior block (no halos) into 1D
            local_block = curr[1:-1, 1:-1].ravel()

            # Root receives all blocks into one big flat buffer
            if rank == 0:
                recvbuf = np.empty(int(counts_flat.sum()), dtype=np.uint8)
            else:
                recvbuf = None

            comm2d.Gatherv(
                local_block,
                (recvbuf, counts_flat, displs_flat, MPI.UNSIGNED_CHAR),
                root=0,
            )

            if rank == 0:
                # Reconstruct full grid from chunks
                for r in range(world_size):
                    row0_r, col0_r, ny_r, nx_r = all_info[r]
                    start = displs_flat[r]
                    end = start + counts_flat[r]
                    block = recvbuf[start:end].reshape(ny_r, nx_r)
                    full[row0_r : row0_r + ny_r, col0_r : col0_r + nx_r] = block

                # Now full is the global grid → save image
                save_frame_png(f"frame_2D_{step:05d}.png", full)

        if args.print_every and (step % args.print_every == 0):
            local_pop = int(curr[1:-1, 1:-1].sum())
            total_pop = comm.allreduce(local_pop, op=MPI.SUM)
            if rank == 0:
                print(f"step {step:6d} population={total_pop}")

    local_pop = int(curr[1:-1, 1:-1].sum())
    total_pop = comm.allreduce(local_pop, op=MPI.SUM)
    if rank == 0:
        print(f"Done. final population={total_pop}")


if __name__ == "__main__":
    main()
