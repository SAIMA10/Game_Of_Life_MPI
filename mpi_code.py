# MPI -Python Script for ring communication using non-blocking send/receive commands
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

left = (rank - 1) % size
right = (rank + 1) % size

send_data_left = np.array([rank, rank + 100], dtype="i")
send_data_right = np.array([rank, rank + 200], dtype="i")

recv_data_left = np.empty_like(send_data_left)
recv_data_right = np.empty_like(send_data_right)

reqs = []

# Receive from left/right
reqs.append(comm.Irecv(recv_data_left, source=left, tag=1))
reqs.append(comm.Irecv(recv_data_right, source=right, tag=0))

# Send to left/right (crossed tags)
reqs.append(comm.Isend(send_data_left, dest=left, tag=0))
reqs.append(comm.Isend(send_data_right, dest=right, tag=1))

MPI.Request.Waitall(reqs)

print(f"Rank {rank} received from left {left}:  {recv_data_left}")
print(f"Rank {rank} received from right {right}: {recv_data_right}")

comm.Barrier()
if rank == 0:
    print("All ranks finished successfully.")
