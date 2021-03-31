#include <cuda_runtime.h> 
#include <cuda.h> 
#include "utils.h"

/*
 * The launcher for your kernels. 
 * This is a single entry point and 
 * all arrays **MUST** be pre-allocated 
 * on device. you must implement all other 
 * kernels in the respective files.
 * */ 
int launch_Jacobi(float *d_U, float *d_U2, int num_rows, int num_cols, int max_iters, float err_thres, float *err_count);