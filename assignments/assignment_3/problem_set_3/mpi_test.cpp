#include <iostream>
#include <cstdio>
#include <cuda.h>
#include <cassert>
#include <string>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <mpi.h>


void checkResult(float *ref, float *gpu, int num_rows, int num_cols, float eps){
     for (int i = 0; i < num_rows; i++){
        for (int j = 0; j < num_cols; j++){
            int location = i * num_cols + j;

            if (ref[location] - gpu[location] > eps){
                std::cerr << "Error at position (" << i << ", " << j << ")\n";

                std::cerr << "Reference:: " << +ref[location] << "\n";
                std::cerr << "GPU:: " << +gpu[location] << "\n";

                exit(1);
            }
        }
    }
    std::cout << "PASSED!\n";
}

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
                return 1;
            }
        }
    }
    return 0;
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
        if (serialLaplacePDEJacobiErrorCheck(U, U2, num_rows, num_cols, err_thres) == 0 || iterations > max_iters){
            return 1;
        }
        // Call a second step of Jacobi
        serialLaplacePDEJacobiSingleStep(U2, U, num_rows, num_cols);
        iterations++;
        // Check for ending conditions
        if (serialLaplacePDEJacobiErrorCheck(U2, U, num_rows, num_cols, err_thres) == 0 || iterations > max_iters){
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
    float *tot_U, *tot_U2, *rank_U, *rank_U2, *buffer;
    float *host_res, *mpi_res;
    int max_iters;
    float err_thres;
    int num_rows, num_cols;
    int rank_rows, rows_per_process;
    int buffer_location;
    MPI_Status status;

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

    // Have processor zero run serial first
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
    }



    // Processor 0 reinitializes Matrix and sends buffers of data out to processors
    if (current_rank == 0){
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

        // Allocate space for its row + 1 ghost row
        rank_U = new float[(rank_rows + 1) * num_cols];
        rank_U2 = new float[(rank_rows + 1) * num_cols];

        // Allocate buffer
        buffer = new float[rows_per_process * num_cols];

        // Iterate over each process
        int row_counter = 0;
        for (int i = 0; i < num_processors; i++){
            // Read data into local rank_U
            if (i == 0){
                for (int j = 0; j < rank_rows; j++){
                    for (int k = 0; k < num_cols; k++){
                        int location = j * num_cols + k;
                        rank_U[location] = tot_U[location];
                        rank_U2[location] = tot_U[location];
                    }
                }
                row_counter = rank_rows;
            // Read data into buffer and then send
            } else {
                buffer_location = 0;
                for (int j = row_counter; j < row_counter+rows_per_process; j++){
                    for (int k = 0; k < num_cols; k++){
                        int location = j * num_cols + k;
                        buffer[buffer_location] = tot_U[location];
                        buffer_location++;
                    }
                }
                row_counter = row_counter + rows_per_process;

                // Send data to corresponding process
                MPI_Send(buffer, rows_per_process*num_cols, MPI_FLOAT, i, i, MPI_COMM_WORLD);
            }
        }

    // Other processors receive data and store to their corresponding matrices
    } else {
        // Receive rows_per_process from rank 0
        MPI_Bcast(&rows_per_process, 1, MPI_INT, 0, MPI_COMM_WORLD);
        rank_rows = rows_per_process;

        // Allocate buffer
        buffer = new float[rows_per_process * num_cols];

        if (current_rank == num_processors - 1){
            // Allocate space for its row + 1 ghost row
            rank_U = new float[(rank_rows + 1) * num_cols];
            rank_U2 = new float[(rank_rows + 1) * num_cols];
        } else {
            // Allocate space for its row + 2 ghost rows
            rank_U = new float[(rank_rows + 2) * num_cols];
            rank_U2 = new float[(rank_rows + 2) * num_cols];
        }

        // Receive rows from processor 0
        MPI_Recv(buffer, rows_per_process*num_cols, MPI_FLOAT, 0, current_rank, MPI_COMM_WORLD, &status);

        // Copy data from buffer into matrix
        buffer_location = 0;
        if (current_rank == num_processors - 1){
            for (int i = 0; i < rank_rows+1; i++){
                for (int j = 0; j < num_cols; j++){
                    int location = i * num_cols + j;
                    if (i == 0){
                        rank_U[location] = 0;
                        rank_U2[location] = 0;
                    } else {
                        rank_U[location] = buffer[buffer_location];
                        rank_U2[location] = buffer[buffer_location];
                        buffer_location++;
                    }
                }
            }
        } else {
            for (int i = 0; i < rank_rows+2; i++){
                for (int j = 0; j < num_cols; j++){
                    int location = i * num_cols + j;
                    if (i == 0 || i == rows_per_process+1){
                        rank_U[location] = 0;
                        rank_U2[location] = 0;
                    } else {
                        rank_U[location] = buffer[buffer_location];
                        rank_U2[location] = buffer[buffer_location];
                        buffer_location++;
                    }
                }
            }
        }
    }

    // sync up all processes
    MPI_Barrier(MPI_COMM_WORLD);

    // Begin Jacobi computation
    /////////////////////////////
    auto mpi_start, mpi_stop, mpi_duration;
    if (current_rank == 0){
        std::cout << "Running MPI Implementation" << std::endl;
        // Start serial timing
        mpi_start = std::chrono::high_resolution_clock::now(); 
    }
    

    // Iterate up to max iterations
    int iterations = 0;
    while (iterations < max_iters){
        // Do data handoff at boundaries
        if (current_rank == 0){
            // Load buffer with true bottom row
            buffer_location = 0;
            for (int j = 0; j < num_cols; j++){
                int location = (rank_rows - 1) * num_cols + j;
                buffer[buffer_location] = rank_U[location];
                buffer_location++;
            }
            // Send bottom row to process 1
            MPI_Send(buffer, num_cols, MPI_FLOAT, current_rank+1, current_rank+1, MPI_COMM_WORLD);

            // Recv top row of process 1
            MPI_Recv(buffer, num_cols, MPI_FLOAT, current_rank+1, current_rank, MPI_COMM_WORLD, &status);

            // Store results as final row of matrix
            buffer_location = 0;
            for (int j = 0; j < num_cols; j++){
                int location = (rank_rows) * num_cols + j;
                rank_U[location] = buffer[buffer_location];
                buffer_location++;
            }
        } else if (current_rank == num_processors - 1){
            // Recv bottom row from previous process
            MPI_Recv(buffer, num_cols, MPI_FLOAT, current_rank-1, current_rank, MPI_COMM_WORLD, &status);

            // Store buffer as top row of matrix
            buffer_location = 0;
            for (int j = 0; j < num_cols; j++){
                int location = (0) * num_cols + j;
                rank_U[location] = buffer[buffer_location];
                buffer_location++;
            }

            // Load buffer with top true row
            buffer_location = 0;
            for (int j = 0; j < num_cols; j++){
                int location = (1) * num_cols + j;
                buffer[buffer_location] = rank_U[location];
                buffer_location++;
            }
            // Send top row to previous process
            MPI_Send(buffer, num_cols, MPI_FLOAT, current_rank-1, current_rank-1, MPI_COMM_WORLD);
        }else {
            // Recv bottom row from previous process 
            MPI_Recv(buffer, num_cols, MPI_FLOAT, current_rank-1, current_rank, MPI_COMM_WORLD, &status);

            // Store buffer as top row of matrix
            buffer_location = 0;
            for (int j = 0; j < num_cols; j++){
                int location = (0) * num_cols + j;
                rank_U[location] = buffer[buffer_location];
                buffer_location++;
            }

            // Load buffer with top true row
            buffer_location = 0;
            for (int j = 0; j < num_cols; j++){
                int location = (1) * num_cols + j;
                buffer[buffer_location] = rank_U[location];
                buffer_location++;
            }
            // Send top row to previous process
            MPI_Send(buffer, num_cols, MPI_FLOAT, current_rank-1, current_rank-1, MPI_COMM_WORLD);


            // Load buffer with bottom true row
            buffer_location = 0;
            for (int j = 0; j < num_cols; j++){
                int location = (rank_rows) * num_cols + j;
                buffer[buffer_location] = rank_U[location];
                buffer_location++;
            }
            // Send bottom row to next process
            MPI_Send(buffer, num_cols, MPI_FLOAT, current_rank+1, current_rank+1, MPI_COMM_WORLD);

            // Recv top row of next process
            MPI_Recv(buffer, num_cols, MPI_FLOAT, current_rank+1, current_rank, MPI_COMM_WORLD, &status);

            // Store results as final row of matrix
            buffer_location = 0;
            for (int j = 0; j < num_cols; j++){
                int location = (rank_rows+1) * num_cols + j;
                rank_U[location] = buffer[buffer_location];
                buffer_location++;
            }
        }

        // sync up all processes
        MPI_Barrier(MPI_COMM_WORLD);

        // Compute one iteration of Jacobi
        if (current_rank == 0){
            // Run single step 
            serialLaplacePDEJacobiSingleStep(rank_U, rank_U2, rank_rows + 1, num_cols);

            // Check for difference for self
            int error_pass = serialLaplacePDEJacobiErrorCheck(rank_U, rank_U2, rank_rows + 1, num_cols, err_thres);
            // Gather differences from all other processes
            for (int i = 1; i < num_processors; i++){
                int error_temp = 0;
                MPI_Recv(&error_temp, 1, MPI_INT, i, i, MPI_COMM_WORLD, &status);
                // Add error to error total
                error_pass = error_pass + error_temp;
            }
            // Determine whether error is less than threshold
            int error_result;
            if (error_pass == 0){
                error_result = 1;
                MPI_Bcast(&error_result, 1, MPI_INT, 0, MPI_COMM_WORLD);
            } else {
                error_result = 0;
                MPI_Bcast(&error_result, 1, MPI_INT, 0, MPI_COMM_WORLD);
            }
            
            // Copy results back into rank_U
            memcpy(rank_U, rank_U2, (rank_rows + 1)*num_cols*sizeof(float));

            // Depending on error results, break out of loop
            if (error_result == 1){
                break;
            }
        } else if (current_rank == num_processors - 1){
            // Run single step
            serialLaplacePDEJacobiSingleStep(rank_U, rank_U2, rank_rows + 1, num_cols);

            // Check for difference for self
            int error_pass = serialLaplacePDEJacobiErrorCheck(rank_U, rank_U2, rank_rows + 1, num_cols, err_thres);
            // Send error to process 0
            MPI_Send(&error_pass, 1, MPI_INT, 0, current_rank, MPI_COMM_WORLD);
            // Receive error results from process 0
            int error_result;
            MPI_Bcast(&error_result, 1, MPI_INT, 0, MPI_COMM_WORLD);

            // Copy results back into rank_U
            memcpy(rank_U, rank_U2, (rank_rows + 1)*num_cols*sizeof(float));

            // Depending on error results, break out of loop
            if (error_result == 1){
                break;
            }
        } else {
            // Run single step
            serialLaplacePDEJacobiSingleStep(rank_U, rank_U2, rank_rows + 2, num_cols);

            // Check for difference for self
            int error_pass = serialLaplacePDEJacobiErrorCheck(rank_U, rank_U2, rank_rows + 1, num_cols, err_thres);
            // Send error to process 0
            MPI_Send(&error_pass, 1, MPI_INT, 0, current_rank, MPI_COMM_WORLD);
            // Receive error results from process 0
            int error_result;
            MPI_Bcast(&error_result, 1, MPI_INT, 0, MPI_COMM_WORLD);

            // Copy results back into rank_U
            memcpy(rank_U, rank_U2, (rank_rows + 2)*num_cols*sizeof(float));

            // Depending on error results, break out of loop
            if (error_result == 1){
                break;
            }
        }

        // sync up all processes and increment iterations
        MPI_Barrier(MPI_COMM_WORLD);
        iterations++;
    }

    // Upon completion, coalesce final results
    if (current_rank == 0){
        // Allocate MPI Results Array
        mpi_res = new float[num_rows * num_cols];

        // Gather differences from all other processes
        int row_counter = 0;
        for (int i = 0; i < num_processors; i++){
            if (i == 0){
                for (int j = 0; j < rank_rows; j++){
                    for (int k = 0; k < num_cols; k++){
                        int location = j * num_cols + k;
                        mpi_res[location] = rank_U[location];
                    }
                }
                row_counter = rank_rows;
            } else {
                // Receive rows from next process in buffer
                MPI_Recv(buffer, rows_per_process*num_cols, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);

                // Write buffer results to results array
                buffer_location = 0;
                for (int j = row_counter; j < row_counter+rows_per_process; j++){
                    for (int k = 0; k < num_cols; k++){
                        int location = j * num_cols + k;
                        mpi_res[location] = buffer[buffer_location];
                        buffer_location++;
                    }
                }
                row_counter = row_counter + rows_per_process;
            }
        }
    } else if (current_rank == num_processors - 1){
        // Put true rows data into buffer
        buffer_location = 0;
        for (int i = 1; i < rank_rows; i++){
            for (int j = 0; j < num_cols; j++){
                int location = i * num_cols + j;
                buffer[buffer_location] = rank_U[location];
                buffer_location++;
            }
        }
        // Send data to corresponding process
        MPI_Send(buffer, rows_per_process*num_cols, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    } else {
        // Put true rows data into buffer
        buffer_location = 0;
        for (int i = 1; i < rank_rows+1; i++){
            for (int j = 0; j < num_cols; j++){
                int location = i * num_cols + j;
                buffer[buffer_location] = rank_U[location];
                buffer_location++;
            }
        }
        // Send data to corresponding process
        MPI_Send(buffer, rows_per_process*num_cols, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }

    // Sync all processes
    MPI_Barrier(MPI_COMM_WORLD);

    // Check for valid answer
    if (current_rank == 0){
        checkResult(host_res, mpi_res, num_rows, num_cols, 2);

        // End serial timing
        mpi_stop = std::chrono::high_resolution_clock::now();
        mpi_duration = std::chrono::duration_cast<std::chrono::microseconds>(mpi_stop - mpi_start); 
        std::cout << "MPI Implementation Completed!" << std::endl;
        std::cout << "MPI Duration: " << mpi_duration.count() << " micro seconds" << std::endl; 
    }

    // Finalize the MPI environment.
    MPI_Finalize();
}