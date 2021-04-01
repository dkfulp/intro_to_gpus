#include <iostream>
#include <cstdio>
#include <cuda.h>
#include <cassert>
#include <string>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <mpi.h>


int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processors
    int num_processors;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processors);

    // Get current processors rank
    int current_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

    std::cout << "Hello from " << current_rank << "\n";

    // Finalize the MPI environment.
    MPI_Finalize();
}