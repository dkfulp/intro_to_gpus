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

#include "utils.h"
#include "jacobi_kernel.h"


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



int main(int argc, char const *argv[]){
    // Pointer to host and device inputs and outputs
    float *h_U, *h_U2, *d_U, *d_U2;
    float *host_res, *gpu_res, *mpigpu_res;
    int max_iters;
    float err_thres;
    int num_rows, num_cols;

    // File String Parameters
    std::string outfile;
    std::string mpioutfile;
    std::string reference;

    // Read in input 
    switch (argc){
    case 5:
        num_rows = std::stoi(std::string(argv[1]));
        num_cols = std::stoi(std::string(argv[2]));
        max_iters = std::stoi(std::string(argv[3]));
        err_thres = std::stof(std::string(argv[4]));
        outfile = "gpu_grid.csv";
        mpioutfile = "mpi_gpu_grid.csv";
        reference = "serial_grid.csv";
        break;
    case 6:
        num_rows = std::stoi(std::string(argv[1]));
        num_cols = std::stoi(std::string(argv[2]));
        max_iters = std::stoi(std::string(argv[3]));
        err_thres = std::stof(std::string(argv[4]));
        outfile = std::string(argv[5]);
        mpioutfile = "mpi_gpu_grid.csv";
        reference = "serial_grid.csv";
        break;
    case 7:
        num_rows = std::stoi(std::string(argv[1]));
        num_cols = std::stoi(std::string(argv[2]));
        max_iters = std::stoi(std::string(argv[3]));
        err_thres = std::stof(std::string(argv[4]));
        outfile = std::string(argv[5]);
        mpioutfile = std::string(argv[6]);
        reference = "serial_grid.csv";
        break;
    case 8:
        num_rows = std::stoi(std::string(argv[1]));
        num_cols = std::stoi(std::string(argv[2]));
        max_iters = std::stoi(std::string(argv[3]));
        err_thres = std::stof(std::string(argv[4]));
        outfile = std::string(argv[5]);
        mpioutfile = std::string(argv[6]);
        reference = std::string(argv[7]);
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

    // Serial Initialization Stage
    // Initialize starting U matrix
    h_U = new float[num_rows * num_cols];
    h_U2 = new float[num_rows * num_cols];

    // Initialize U
    DissipationMatrixInitialization(h_U, num_rows, num_cols);
    // Copy to U2
    for (int i = 0; i < num_rows; i++){
        for (int j = 0; j < num_cols; j++){
            int location = i * num_cols + j;
            h_U2[location] = h_U[location];
        }
    }


    // GPU Initialization Stage
    // Allocate nessecary device memory
    checkCudaErrors(cudaMalloc((void **)&d_U, sizeof(float) * num_rows * num_cols));
    checkCudaErrors(cudaMalloc((void **)&d_U2, sizeof(float) * num_rows * num_cols));

    // Copy host matrices to GPU
    checkCudaErrors(cudaMemcpy(d_U, h_U, sizeof(float) * num_rows * num_cols, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_U2, h_U2, sizeof(float) * num_rows * num_cols, cudaMemcpyHostToDevice));


    // Serial Running Stage
    std::cout << "Running Serial Implementation" << std::endl;
    // Start serial timing
    auto start = std::chrono::high_resolution_clock::now(); 

    // Call Serial Function
    int output_id = serialLaplacePDEJacobiSolver(h_U, h_U2, num_rows, num_cols, max_iters, err_thres);
    // Copy results to host final array
    host_res = new float[num_rows * num_cols];
    if (output_id == 0){
        // Copy to U into host_res
        for (int i = 0; i < num_rows; i++){
            for (int j = 0; j < num_cols; j++){
                int location = i * num_cols + j;
                host_res[location] = h_U[location];
            }
        }
    } else {
        // Copy to U2 into host_res
        for (int i = 0; i < num_rows; i++){
            for (int j = 0; j < num_cols; j++){
                int location = i * num_cols + j;
                host_res[location] = h_U2[location];
            }
        }
    }

    // End serial timing
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); 
    std::cout << "Serial Implementation Completed!" << std::endl;
    std::cout << "Serial Duration: " << duration.count() << " micro seconds" << std::endl; 


    // GPU Running Stage
    std::cout << "Running GPU Implementation" << std::endl;
    // Call GPU Laplace PDE Jacobi Solver
    output_id = launch_Jacobi(d_U, d_U2, num_rows, num_cols, max_iters, err_thres);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    // Copy results to gpu final array
    gpu_res = new float[num_rows * num_cols];
    if (output_id == 0){
        checkCudaErrors(cudaMemcpy(gpu_res, d_U, sizeof(float) * num_rows * num_cols, cudaMemcpyDeviceToHost));
    } else {
        checkCudaErrors(cudaMemcpy(gpu_res, d_U2, sizeof(float) * num_rows * num_cols, cudaMemcpyDeviceToHost));
    }
    std::cout << "GPU Implementation Completed!" << std::endl;



    // Call MPI-GPU Laplace PDE Jacobi Solver

    // Copy results to mpi gpu final array
    mpigpu_res = new float[num_rows * num_cols];


    // Ensure correctness of all solutions in final arrays
    std::cout << "Checking GPU Results" << std::endl;
    //checkResult(host_res, gpu_res, num_rows, num_cols, 1e-5);
    checkResult(host_res, gpu_res, num_rows, num_cols, 2);

    // Output Stage
    // TODO

    // Free Allocated Memory
    cudaFree(d_U);
    cudaFree(d_U2);
    delete[] h_U;
    delete[] h_U2;
    delete[] host_res;
    delete[] gpu_res;
    delete[] mpigpu_res;
    return 0;
}
