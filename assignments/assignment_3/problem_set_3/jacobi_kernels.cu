#include "./jacobi_kernel.h" 

#define BLOCK 32


///////////////////////////
///// Jacobi Functions ////
///////////////////////////
__global__ 
void Jacobi_Single_Step(float *d_U, float *d_U2, int num_rows, int num_cols){
    // Determine location of target cell in global memory
    // Shift every thread down and forward one to ensure not working on edge pixels
    int gl_row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int gl_col = blockIdx.x * blockDim.x + threadIdx.x + 1;

    // Ensure the target cell is valid and not on the edge
    if (gl_col < (num_cols - 1) && gl_row < (num_rows - 1)){
        // Get cells location
        int location = gl_row * num_cols + gl_col;

        // Get neighboring cell locations
        int up_location = (gl_row - 1) * num_cols + gl_col;
        int down_location = (gl_row + 1) * num_cols + gl_col;
        int left_location = gl_row * num_cols + (gl_col - 1);
        int right_location = gl_row * num_cols + (gl_col + 1);

        // Calculate weighted average
        d_U2[location] = 0.25 * (d_U[up_location] + d_U[down_location] + d_U[left_location] + d_U[right_location]);
    }
}



///////////////////////////
///// Helper Functions ////
///////////////////////////
__global__ 
void Jacobi_Error_Check(float *d_U, float *d_U2, int num_rows, int num_cols, float err_thres, int error_count){
    // Determine location of target cell in global memory
    // Shift every thread down and forward one to ensure not working on edge pixels
    int gl_row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int gl_col = blockIdx.x * blockDim.x + threadIdx.x + 1;

    // Ensure the target cell is valid and not on the edge
    if (gl_col < (num_cols - 1) && gl_row < (num_rows - 1)){
        // Get cells location
        int location = gl_row * num_cols + gl_col;

        // Check to see if difference is below threshold
        if (d_U[location] - d_U2[location] > err_thres){
            error_count = error_count + 1;
        }
    }
}


///////////////////////////
///// Running Kernels /////
///////////////////////////
int launch_Jacobi(float *d_U, float *d_U2, int num_rows, int num_cols, int max_iters, float err_thres){
    // Set grid and block dimensions
    dim3 grid(std::ceil((float)(num_cols - 2)/(float)BLOCK),std::ceil((float)(num_rows - 2)/(float)BLOCK),1);
    dim3 block(BLOCK, BLOCK, 1);

    // Repeatedly call Jacobi kernel 
    int iterations = 0;
    int error_count = 0;

    // Go for maximum number of iterations
    while(true){
        // Call a single step of Jacobi
        Jacobi_Single_Step<<<grid,block>>>(d_U, d_U2, num_rows, num_cols);
        cudaDeviceSynchronize();
	    checkCudaErrors(cudaGetLastError());
        // Increment iterations
        iterations++;
        // Check for ending conditions
        Jacobi_Error_Check<<<grid,block>>>(d_U, d_U2, num_rows, num_cols, err_thres, error_count);
        cudaDeviceSynchronize();
	    checkCudaErrors(cudaGetLastError());
        if (error_count == 0 || iterations > max_iters){
            // Return flag stating U2 has final information
            return 1;
        }
        // Reset error count
        error_count = 0

        // Call a second step of Jacobi
        Jacobi_Single_Step<<<grid,block>>>(d_U2, d_U, num_rows, num_cols);
        cudaDeviceSynchronize();
	    checkCudaErrors(cudaGetLastError());
        // Increment iterations
        iterations++;
        // Check for ending conditions
        Jacobi_Error_Check<<<grid,block>>>(d_U2, d_U, num_rows, num_cols, err_thres, error_count);
        cudaDeviceSynchronize();
	    checkCudaErrors(cudaGetLastError());
        if (error_count == 0 || iterations > max_iters){
            // Return flag stating U has final information
            return 0;
        }
        // Reset error count
        error_count = 0
    }
}