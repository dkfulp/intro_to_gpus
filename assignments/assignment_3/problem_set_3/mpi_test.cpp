#include <iostream>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <string>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <mpi.h>


int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    printf("Hello world! from rank %d"
           " out of %d processors\n",
           world_rank, world_size);

    // Finalize the MPI environment.
    MPI_Finalize();
}