from mpi4py import MPI

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

destino = (rank + 1) % size
origen = (rank - 1) % size

for generacion in range(4):

    print(
        f"Isla {rank}: iniciando intercambio {generacion + 1}",
        flush=True
    )

    recibido = comm.sendrecv(
        rank,
        dest=destino,
        sendtag=0,
        source=origen,
        recvtag=0
    )

    print(
        f"Isla {rank}: recibió {recibido} correctamente",
        flush=True
    )

comm.Barrier()

if rank == 0:
    print("PRUEBA MPI COMPLETADA", flush=True)

