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

int main(int argc, char *argv[]){

    int id;
    int ierr;
    int p;
    
    ierr = MPI_Init( &argc, &argv );

    if (ierr != 0){
        std::cout << "\n";
        std::cout << "HELLO_MPI - Fatal error!\n";
        std::cout << "  MPI_Init returned nonzero ierr.\n";
        exit ( 1 );
    }

    // Get number of processors
    ierr = MPI_Comm_size ( MPI_COMM_WORLD, &p );

    // Get processor id
    ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &id );

    // Print Statement
    std::cout << "P" << id << ":    'Hello, world!'\n";

    // Finalize MPI
    MPI_Finalize ( );
}