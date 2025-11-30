#!/usr/bin/env python
"""
With 2D decomposition: P x Q grid of ranks (P rows of ranks, Q columns of ranks).
Each rank owns a sub-block: shape ≈ (local_ny + 2, local_nx + 2) (now both row and column halos).
Exchange halos in four directions: up, down, left, right (and optionally corners).
"""
import argparse

# to parse arguments from command line
import numpy as np

# multi-dimensional arrays (grid),
# vectorized computing (neighbouring counting), reshape, roll, etc
from mpi4py import MPI

# halo exchanges, cartesian topology (2D), rank and size info, gather % scatter operations, sync, non-blocking send/recv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# added for the images, Agg means do not open windows and render images in PNGs directly, otherwise crashes on cluster


def save_frame_png(path, arr):
    plt.figure(figsize=(6, 6))
    # arr is the global grid of the shape (ny, nx) gathered on rank 0, numpy array
    plt.imshow(arr, cmap="hot", vmin=0, vmax=1, interpolation="nearest")
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
    p.add_argument("--init", choices=["gosper", "random"], default="gosper")
    p.add_argument("--print-every", type=int, default=0)
    return p.parse_args()


def decompose_2d(ny, nx, p_rows, p_cols, prow, pcol):
    # rows, same as 1D
    base_ny = ny // p_rows
    rem_ny = ny % p_rows
    local_ny = base_ny + (1 if prow < rem_ny else 0)
    row0 = prow * base_ny + min(prow, rem_ny)

    # cols
    base_nx = nx // p_cols
    rem_nx = nx % p_cols
    local_nx = base_nx + (1 if pcol < rem_nx else 0)
    col0 = pcol * base_nx + min(pcol, rem_nx)
    # same logic as rows but with cols
    """
    row0 - Starting global row index this rank owns
    col0 - Starting global col index this rank owns
    local_ny - no of rows assigned to this rank
    local_nx - no of cols assigned to this rank
    """

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

    # Convert local offsets → global absolute coordinates
    global_rows = pat_row0 + gosper[:, 0]
    global_cols = pat_col0 + gosper[:, 1]

    # wrap on torus
    global_rows %= ny
    global_cols %= nx
    # If the pattern extends beyond the right edge, it wraps to the left.
    # If the pattern extends beyond the bottom edge, it wraps to the top.

    # pattern cells fall into this block
    in_block = (
        (global_rows >= row0)
        & (global_rows < row0 + local_ny)
        & (global_cols >= col0)
        & (global_cols < col0 + local_nx)
    )
    if not np.any(in_block):
        return

    # map global → local (shift by block origin, +1 for halos) after figuring out which cells are actaully inside the region
    local_rows = (global_rows[in_block] - row0) + 1
    local_cols = (global_cols[in_block] - col0) + 1

    local_grid[local_rows, local_cols] = 1
    # Each rank places the alive pattern cells into the correct positions inside its own piece of the grid.


def life_step(curr, nxt):
    # curr/nxt: shape (local_ny+2, local_nx+2) with halos

    center = curr[1:-1, 1:-1]
    # removes the halo rows/columns, center contains only the real cells owned by this rank.

    nbrs = (
        curr[0:-2, 0:-2]
        # Upper-left neighbors - Rows shifted up by 1, columns shifted left by 1
        + curr[0:-2, 1:-1]
        # Upper neighbors - Rows above the interior rows, same column alignment.
        + curr[0:-2, 2:]
        # Upper-right neighbors
        + curr[1:-1, 0:-2]
        # Left neighbors, Interior rows, columns shifted left
        + curr[1:-1, 2:]
        # right neighbours, Interior rows, columns shifted right.
        + curr[2:, 0:-2]
        # lower-left neighbours, Rows below the center, columns left.
        + curr[2:, 1:-1]
        # lower neighbours, Rows below, same columns.
        + curr[2:, 2:]
        # Lower-right neighbors, Rows below, columns right.
    )

    nxt[1:-1, 1:-1] = ((nbrs == 3) | ((center == 1) & (nbrs == 2))).astype(np.uint8)


def main():
    args = parse_args()
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # size = comm.Get_size()

    comm = MPI.COMM_WORLD
    # world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    # Make a 2D process grid: dims[0] * dims[1] == world_size
    dims = MPI.Compute_dims(world_size, 2)  # e.g. 4→[2,2], 8→[2,4] etc.
    # 2 process rows, 4 process cols
    periods = [True, True]
    # MPI cartesian support periodic boundries - grid edges wrap wround
    # periods[0] = True → wrap up/down
    # periods[1] = True → wrap left/right
    comm2d = comm.Create_cart(dims, periods=periods, reorder=True)

    rank = comm2d.Get_rank()
    # Each process now gets a new rank ID inside the 2D grid communicator.
    coords = comm2d.Get_coords(rank)  # (prow, pcol)
    p_rows, p_cols = dims  # number of ranks in each direction
    prow, pcol = coords  # this rank's coordinates in the grid

    nx, ny = args.nx, args.ny

    row0, col0, local_ny, local_nx = decompose_2d(ny, nx, p_rows, p_cols, prow, pcol)

    # each rank own a set of local rows, cols (array) of its block, so it'll look like [4,0,4,4]
    local_info = np.array([row0, col0, local_ny, local_nx], dtype="i")
    all_info = None  # rank0 to gather all ranks for visualization, others are None
    if rank == 0:
        all_info = np.empty((world_size, 4), dtype="i")
        # Rank 0 allocates a 2D array to hold one local_info entry for each rank.

    comm2d.Gather(local_info, all_info, root=0)
    # This collects the local_info from every rank into all_info on rank 0.
    """
    all_info = [
    [row0(rank0), col0(rank0), local_ny(r0), local_nx(r0)],
    [row0(rank1), col0(rank1), local_ny(r1), local_nx(r1)],
    ...
    ]
    """
    if rank == 0:
        block_sizes = all_info[:, 2] * all_info[:, 3]
        # local_ny * local_nx per rank -> all_info[:,2] = local_ny for each rank, all_info[:,3] = local_nx for each rank
        # number of interior cells on each rank
        counts_flat = block_sizes.astype("i")
        # requirement from Gather/ Gatherv to be integer (32-bit integer)
        displs_flat = np.zeros(world_size, dtype="i")
        # This will be filled with the offsets where each rank’s data starts inside the final flattened buffer.
        displs_flat[1:] = np.cumsum(counts_flat[:-1])
        # cumsum(counts_flat[:-1]) = [16, 32, 48, 64, 80]
        # displs_flat = [0, 16, 32, 48, 64, 80]
        # Rank 0 → place at offset 0
        # Rank 1 → place at offset 16
        # Rank 2 → at offset 32
        # Rank 3 → at offset 48

    else:  # for other ranks
        counts_flat = None
        displs_flat = None

    # allocate with halos on BOTH sides
    curr = np.zeros((local_ny + 2, local_nx + 2), dtype=np.uint8)
    # uint8 - binary, 0 or 1
    nxt = np.zeros_like(curr)
    # life_step(curr, nxt) writes the next generation into nxt -> we swap -> Then nxt becomes a clean buffer for the next step.
    # This avoids overwriting curr while you're still reading from it.

    full = np.empty((ny, nx), dtype=np.uint8) if rank == 0 else None
    # Rank 0 allocates a large global array to assemble the complete grid. -> generate the image

    # Seed pattern by global coordinates; only owning ranks place cells,
    # ensures that any future random actions will be reproducible
    np.random.seed(args.seed)

    if args.init == "random":
        curr[1:-1, 1:-1] = np.random.randint(0, 2, (local_ny, local_nx))
    elif args.init == "gosper":
        seed_gosper(curr, ny, nx, row0, col0, local_ny, local_nx)

    comm2d.Barrier()
    t0 = MPI.Wtime()  # start timer (seconds since some MPI-defined time)

    up, down = comm2d.Shift(direction=0, disp=1)  # row neighbors, vertical neighbours
    left, right = comm2d.Shift(direction=1, disp=1)
    # column neighbors, horizontal neighbours

    """
    PNG : Rank-ID visualization: which rank owns which cells
    """

    # Local block of rank IDs (interior only, no halos)
    local_ids = np.full((local_ny, local_nx), rank, dtype=np.int32)

    # Flatten for Gatherv
    local_ids_flat = local_ids.ravel()

    if rank == 0:
        recv_ids = np.empty(int(counts_flat.sum()), dtype=np.int32)
    else:
        recv_ids = None

    comm2d.Gatherv(
        local_ids_flat,
        (recv_ids, counts_flat, displs_flat, MPI.INT),
        root=0,
    )

    if rank == 0:
        # Reconstruct global rank-ID grid
        full_ids = np.empty((ny, nx), dtype=np.int32)
        for r in range(world_size):
            row0_r, col0_r, ny_r, nx_r = all_info[r]
            start = displs_flat[r]
            end = start + counts_flat[r]
            block = recv_ids[start:end].reshape(ny_r, nx_r)
            full_ids[row0_r : row0_r + ny_r, col0_r : col0_r + nx_r] = block

        # Plot: each rank gets a different color
        plt.figure(figsize=(6, 6))
        plt.imshow(full_ids, cmap="tab20", interpolation="nearest")
        plt.colorbar(label="MPI rank")
        plt.title("2D Decomposition: Rank ownership")
        plt.axis("off")
        plt.tight_layout(pad=0.1)
        plt.savefig("decomposition_ranks.png", dpi=150)
        plt.close()

    """"""

    for step in range(args.steps):
        # Halo exchange using your exact pattern: Irecv/Irecv + Isend/Isend + Waitall
        reqs = []  # store all non-blocking operations here so we can Waitall() later.

        # vertical halos (up/down)
        reqs.append(comm2d.Irecv(curr[0, 1:-1], source=up, tag=10))  # top halo
        # Tag = 10 so it matches the send-down from the UP rank.
        # Receive from UP neighbor.
        # Store into your top halo row: curr[0, 1:-1].
        reqs.append(comm2d.Irecv(curr[-1, 1:-1], source=down, tag=11))  # bottom halo

        reqs.append(comm2d.Isend(curr[1, 1:-1], dest=up, tag=11))
        # Your first interior row (just below the top halo) goes to the UP neighbor’s bottom halo
        reqs.append(comm2d.Isend(curr[-2, 1:-1], dest=down, tag=10))
        # Your last interior row goes to DOWN neighbor’s top halo.

        # horizontal halos (left/right)
        # Columns are not contiguous in memory → we must .copy(), rows are cause of row major order in numpy
        send_left = curr[1:-1, 1].copy()  # your first interior column
        send_right = curr[1:-1, -2].copy()  # your last interior column
        recv_left = np.empty_like(send_left)
        recv_right = np.empty_like(send_right)

        reqs.append(comm2d.Irecv(recv_left, source=left, tag=20))
        # From LEFT neighbor. Data is placed into recv_left
        reqs.append(comm2d.Irecv(recv_right, source=right, tag=21))
        reqs.append(comm2d.Isend(send_left, dest=left, tag=21))
        reqs.append(comm2d.Isend(send_right, dest=right, tag=20))

        MPI.Request.Waitall(reqs)
        # ensures its uptodate for the next computation

        # fill halo columns
        curr[1:-1, 0] = recv_left  # left halo
        curr[1:-1, -1] = recv_right  # right halo
        # ensures halos are in place before compute.

        # Compute next generation on interior
        life_step(curr, nxt)
        curr, nxt = nxt, curr

        # gather & save frame on rank 0 every print-every steps
        if args.print_every and (step % args.print_every == 0):
            # Flatten our interior block (no halos) into 1D
            local_block = curr[1:-1, 1:-1].ravel()
            # curr[1:-1, 1:-1] = interior grid
            # .ravel() turns that 2D block into a 1D array of length

            # Root receives all blocks into one big flat buffer
            if rank == 0:
                # total number of cells in the whole global grid
                recvbuf = np.empty(int(counts_flat.sum()), dtype=np.uint8)
            else:
                recvbuf = None

            comm2d.Gatherv(
                local_block,
                (recvbuf, counts_flat, displs_flat, MPI.UNSIGNED_CHAR),
                root=0,
            )
            # MPI uses this map to assemble a variable-size composite buffer.

            if rank == 0:
                # Reconstruct full grid from chunks
                for r in range(world_size):
                    row0_r, col0_r, ny_r, nx_r = all_info[r]
                    # row0_r, col0_r where that rank’s block starts in the global grid
                    # block size (ny_r, nx_r)
                    start = displs_flat[r]
                    # Where in recvbuf this rank’s data starts
                    end = start + counts_flat[r]
                    # Where it ends
                    block = recvbuf[start:end].reshape(ny_r, nx_r)
                    # Take that 1D chunk and reshape it back into a 2D block
                    full[row0_r : row0_r + ny_r, col0_r : col0_r + nx_r] = block

                # Now full is the global grid → save image
                save_frame_png(f"frame_2D_{step:05d}.png", full)

        if args.print_every and (step % args.print_every == 0):
            local_pop = int(curr[1:-1, 1:-1].sum())
            # all live cells in the interior region
            total_pop = comm.allreduce(local_pop, op=MPI.SUM)
            # MPI collective communication
            if rank == 0:
                print(f"step {step:6d} population={total_pop}")

    comm.Barrier()  # ensure all ranks finished the loop
    t1 = MPI.Wtime()
    loop_time = t1 - t0

    local_pop = int(curr[1:-1, 1:-1].sum())
    total_pop = comm.allreduce(local_pop, op=MPI.SUM)
    if rank == 0:
        print(f"Done. final population={total_pop}")
        print(f"RANKS={world_size} STEPS={args.steps} TIME_LOOP={loop_time:.6f} s ")


if __name__ == "__main__":
    main()
