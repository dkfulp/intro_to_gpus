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


void DissipationMatrixInitialization(float *U, int num_rows, int num_cols){
    for (int i = 0; i < num_rows; i++){
        for (int j = 0; j < num_cols; j++){
            int location = i * num_cols + j;

            // Set top and left to heat sources
            if (i == 0 || j == 0){
                U[location] = 100;
            // Set bottom and right to heat sinks
            } else if (i == (num_rows - 1) || j == (num_cols - 1)){
                U[location] = -100;
            // Set others
            } else {
                U[location] = 0;
            }
        }
    }
}

int serialLaplacePDEJacobiErrorCheck(float *U, float *U2, int num_rows, int num_cols, float err_thres){
    // Iterate over rows
    for (int i = 1; i < num_rows-1; i++){
        // Iterate over columns
        for (int j = 1; j < num_cols-1; j++){
            // Get cells location
            int location = i * num_cols + j;

            // Check to see if difference is below threshold
            if (U[location] - U2[location] > err_thres){
                return 0;
            }
        }
    }
    return 1;
}

void serialLaplacePDEJacobiSingleStep(float *U, float *U2, int num_rows, int num_cols){
    // Iterate over rows
    for (int i = 1; i < num_rows-1; i++){
        // Iterate over columns
        for (int j = 1; j < num_cols-1; j++){
            // Get cells location
            int location = i * num_cols + j;

            // Get neighboring cell locations
            int up_location = (i - 1) * num_cols + j;
            int down_location = (i + 1) * num_cols + j;
            int left_location = i * num_cols + (j - 1);
            int right_location = i * num_cols + (j + 1);

            // Calculate weighted average
            float average = 0.25 * (U[up_location] + U[down_location] + U[left_location] + U[right_location]);

            // Update cells value using surrounding cells
            U2[location] = average;
        }
    }
}

int serialLaplacePDEJacobiSolver(float *U, float *U2, int num_rows, int num_cols, int max_iters, float err_thres){
    int iterations = 0;

    // Go for maximum number of iterations
    while(true){
        // Call a single step of Jacobi
        serialLaplacePDEJacobiSingleStep(U, U2, num_rows, num_cols);
        iterations++;
        // Check for ending conditions
        if (serialLaplacePDEJacobiErrorCheck(U, U2, num_rows, num_cols, err_thres) == 1 || iterations > max_iters){
            return 1;
        }
        // Call a second step of Jacobi
        serialLaplacePDEJacobiSingleStep(U2, U, num_rows, num_cols);
        iterations++;
        // Check for ending conditions
        if (serialLaplacePDEJacobiErrorCheck(U2, U, num_rows, num_cols, err_thres) == 1 || iterations > max_iters){
            return 0;
        }
    }
}

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    // Get the number of processors
    int num_processors;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processors);
    // Get current processors rank
    int current_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

    // Pointer to host and device inputs and outputs
    float *tot_U, *tot_U2, *rank_U, *rank_U2;
    float *host_res, *mpi_res;
    int max_iters;
    float err_thres;
    int num_rows, num_cols;
    int rank_rows, rows_per_process;

    // File String Parameters
    std::string outfile;
    std::string reference;

    // Read in input 
    switch (argc){
    case 5:
        num_rows = std::stoi(std::string(argv[1]));
        num_cols = std::stoi(std::string(argv[2]));
        max_iters = std::stoi(std::string(argv[3]));
        err_thres = std::stof(std::string(argv[4]));
        outfile = "mpi_grid.csv";
        reference = "serial_grid.csv";
        break;
    case 6:
        num_rows = std::stoi(std::string(argv[1]));
        num_cols = std::stoi(std::string(argv[2]));
        max_iters = std::stoi(std::string(argv[3]));
        err_thres = std::stof(std::string(argv[4]));
        outfile = std::string(argv[5]);
        reference = "serial_grid.csv";
        break;
    case 7:
        num_rows = std::stoi(std::string(argv[1]));
        num_cols = std::stoi(std::string(argv[2]));
        max_iters = std::stoi(std::string(argv[3]));
        err_thres = std::stof(std::string(argv[4]));
        outfile = std::string(argv[5]);
        reference = std::string(argv[6]);
        break;
    default:
        std::cerr << "Usage ./pde <rows> <cols> <max_iters> <err_thres> <gpu_grid> <mpi_gpu_grid> <serial_grid> \n";
        exit(1);
    }

    // Check for validity
    if (num_rows <= 0 || num_cols <= 0 || max_iters <= 0 || err_thres <= 0){
        std::cerr << "Please enter valid values for rows, cols, max_iters, and err_thres before trying again...\n";
        exit(1);
    }

    // Have processor zero initialize total matrix and run serial implementation
    if (current_rank == 0){
        // Run serial implementation
        ///////////////////////////////        
        // Initialize total U and resulting U2 matrix
        tot_U = new float[num_rows * num_cols];
        tot_U2 = new float[num_rows * num_cols];
        // Initialize U
        DissipationMatrixInitialization(tot_U, num_rows, num_cols);
        // Copy to U2
        for (int i = 0; i < num_rows; i++){
            for (int j = 0; j < num_cols; j++){
                int location = i * num_cols + j;
                tot_U2[location] = tot_U[location];
            }
        }

        // Serial Running Stage
        std::cout << "Running Serial Implementation" << std::endl;
        // Start serial timing
        auto start = std::chrono::high_resolution_clock::now(); 

        // Call Serial Function
        int output_id = serialLaplacePDEJacobiSolver(tot_U, tot_U2, num_rows, num_cols, max_iters, err_thres);
        // Copy results to host final array
        host_res = new float[num_rows * num_cols];
        if (output_id == 0){
            // Copy to U into host_res
            for (int i = 0; i < num_rows; i++){
                for (int j = 0; j < num_cols; j++){
                    int location = i * num_cols + j;
                    host_res[location] = tot_U[location];
                }
            }
        } else {
            // Copy to U2 into host_res
            for (int i = 0; i < num_rows; i++){
                for (int j = 0; j < num_cols; j++){
                    int location = i * num_cols + j;
                    host_res[location] = tot_U2[location];
                }
            }
        }

        // End serial timing
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); 
        std::cout << "Serial Implementation Completed!" << std::endl;
        std::cout << "Serial Duration: " << duration.count() << " micro seconds" << std::endl; 


        // Run MPI Implementation
        ////////////////////////////
        // Reinitialize U
        DissipationMatrixInitialization(tot_U, num_rows, num_cols);
        // Copy to U2
        for (int i = 0; i < num_rows; i++){
            for (int j = 0; j < num_cols; j++){
                int location = i * num_cols + j;
                tot_U2[location] = tot_U[location];
            }
        }

        // Determine the number of rows to send to each other process
        rows_per_process = std::floor(num_rows/ num_processors);
        // Broadcast rows_per_process
        MPI_Bcast(&rows_per_process, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Determine number of rows processor zero will do
        rank_rows = num_rows % num_processors + rows_per_process;

        /// Allocate space for its rows + 1
        std::cout << "Process 0: " << rank_rows << "\n";
        

        // Read nessecary self rows into rank_U and rank_U2
        // Use MPI_Send buffer to send the rest of the rows to other processes

    } else {
        // Receive rows_per_process from rank 0
        MPI_Bcast(&rows_per_process, 1, MPI_INT, 0, MPI_COMM_WORLD);
        rank_rows = rows_per_process;

        std::cout << "Process 0: " << rank_rows << "\n";

        if (current_rank == num_processors - 1){
            // Allocate space for their rows + 1
        } else {
            // Allocate space for their rows + 2
        }

        // Receive rows from rank zero
    }

    // sync up all processes
    MPI_Barrier(MPI_COMM_WORLD);

    // Begin Jacobi computation
    /////////////////////////////

    // Start timing
    // TODO

    // Do data handoff at boundaries
    if (current_rank == 0){
        // MPI_Send bottom row to process 1
        // MPI_Recv top row of process 1 as bottom row of data
    } else if (current_rank == num_processors - 1){
        // MPI_Recv bottom row from previous process
        // MPI_Send top row to previous process
    }else {
        // MPI_Recv bottom row from previous process
        // MPI_Send top row to previous process
        // MPI_Send bottom row to next process
        // MPI_Recv top row of next process as bottom row of data
    }

    // Compute one iterations of Jacobi

    // Check for ending condition with error threshold

    // Repeat process until end

        // If ending conditions met, all processes send data back to process 0

        // Process 0 collects data and stores to variable

        // Process 0 also checks against serial before ending
    


    // Finalize the MPI environment.
    MPI_Finalize();
}