#include "./gaussian_kernel.h" 

#define BLOCK 32


// gaussianBlurGlobal:
// Kernel that computes a gaussian blur over a single RGB channel. 
// This implementation in specific does not use shared memory to 
// improve performance.
__global__ 
void gaussianBlurGlobal(unsigned char *d_in, unsigned char *d_out, const int num_rows, const int num_cols, float *d_filter, const int filterWidth){
        // Initialize row and column operators
        int in_row, in_col;

        // Determine location of target pixel in global memory
        int gl_row = blockIdx.y * blockDim.y + threadIdx.y;
        int gl_col = blockIdx.x * blockDim.x + threadIdx.x;

        // Ensure the target pixel is valid
        if (gl_col < num_cols && gl_row < num_rows){
                // Given the filter width, determine the correct row and col offsets
                int blur_offset = ((filterWidth-1)/2);

                // Setup loop variables
                int blur_sum = 0;
                int filter_pos = 0;

                // Iterate from the furthest back row to the furthest forward row
                for (in_row = gl_row - blur_offset; in_row <= gl_row + blur_offset; in_row++){
                        // Iterate from the furthest back col to the furthest forward col
                        for (in_col = gl_col - blur_offset; in_col <= gl_col + blur_offset; in_col++){
                                // Ensure target blur pixel location is valid
                                if (in_row >= 0 && in_row < num_rows && in_col >= 0 && in_col < num_cols){
                                        // Get target blur pixel offset
                                        int pixel_offset = in_row * num_cols + in_col;

                                        // Multiply current filter location by target pixel and add to running sum
                                        blur_sum += (int)( (float)d_in[pixel_offset] * d_filter[filter_pos] );
                                }
                                // Always increment filter location
                                filter_pos++;
                        }
                }
                // Store results in the correct location of the output array
                int result_offset = gl_row * num_cols + gl_col;
                d_out[result_offset] = (unsigned char)blur_sum;
        }
} 


// gaussianBlurShared:
// Kernel that computes a gaussian blur over a single RGB channel. 
// This implementation in specific uses shared memory to reduce the 
// number of accesses to global memory to improve performance.
 __global__ 
 void gaussianBlurShared(unsigned char *d_in, unsigned char *d_out, const int num_rows, const int num_cols, float *d_filter, const int filterWidth){
        // Create shared memory input and array
        __shared__ unsigned char input_pixels[BLOCK*BLOCK];

        // Get location in global memory 
        int gl_row = blockIdx.y * blockDim.y + threadIdx.y;
        int gl_col = blockIdx.x * blockDim.x + threadIdx.x;

        // Get location in shared memory
        int sh_row = threadIdx.y;
        int sh_col = threadIdx.x;

        // Ensure target working pixel is valid
        if (gl_col < num_cols && gl_row < num_rows){
                // Load shared memory values for all interal pixels from global memory
                int global_offset = gl_row * num_cols + gl_col;
                int shared_offset = sh_row * blockDim.x + sh_col;
                input_pixels[shared_offset] = d_in[global_offset];
        
                // Make sure all threads have loaded before starting computation
                __syncthreads();

                // Setup loop variables
                int in_row, in_col;
                int in_gl_row, in_gl_col;
                int blur_sum = 0;
                int filter_pos = 0;

                // Given the filter width, determine the correct row and col offsets
                int blur_offset = ((filterWidth-1)/2);

                // Iterate from the furthest back row to the furthest forward row
                for (in_row = sh_row - blur_offset; in_row <= sh_row + blur_offset; in_row++){
                        // Iterate from the furthest back col to the furthest forward col
                        for (in_col = sh_col - blur_offset; in_col <= sh_col + blur_offset; in_col++){
                                // Target Pixel is In Shared Memory
                                if (in_row >= 0 && in_row < blockDim.y && in_col >= 0 && in_col < blockDim.x){
                                        // Get target blur pixel offset
                                        int shared_offset = in_row * blockDim.x + in_col;

                                        // Multiply current filter location by target pixel and add to running sum
                                        blur_sum += (int)( (float)input_pixels[shared_offset] * d_filter[filter_pos] );
                                // Target Pixel is Not In Shared Memory
                                } else {
                                        // Ensure target pixel global location is valid
                                        in_gl_row = blockIdx.y * blockDim.y + in_row;
                                        in_gl_col = blockIdx.x * blockDim.x + in_col;
                                        if (in_gl_row >= 0 && in_gl_row < num_rows && in_gl_col >= 0 && in_gl_col < num_cols){
                                                // Get target blur pixel offset
                                                int global_offset = in_gl_row * num_cols + in_gl_col;

                                                // Multiply current filter location by target pixel and add to running sum
                                                blur_sum += (int)( (float)d_in[global_offset] * d_filter[filter_pos] );
                                        }
                                }
                                // Always increment filter location
                                filter_pos++;
                        }
                }

                // Make sure all threads have finished computation
                __syncthreads();

                // Store results in the correct location of the output array
                int result_offset = gl_row * num_cols + gl_col;
                d_out[result_offset] = (unsigned char)blur_sum;
        } 
} 
 



// separateChannels:
// Kernel that splits an RGBA uchar4 array into 3 seperate unsigned char 
// arrays.
__global__ 
void separateChannels(uchar4 *d_imrgba, unsigned char *d_r, unsigned char *d_g, unsigned char *d_b, const int num_rows, const int num_cols){
        // Determine location of target pixel in global memory
        int gl_row = blockIdx.y * blockDim.y + threadIdx.y;
        int gl_col = blockIdx.x * blockDim.x + threadIdx.x;

        // Ensure the target pixel is valid
        if (gl_col < num_cols && gl_row < num_rows){
                // Get pixel location
                int pixel_offset = gl_row * num_cols + gl_col;
                // Get corresponding rgba pixel at this location
                uchar4 rgba_pixel = d_imrgba[pixel_offset];
                // Save each pixel element to correct array
                d_r[pixel_offset] = rgba_pixel.x;
                d_g[pixel_offset] = rgba_pixel.y;
                d_b[pixel_offset] = rgba_pixel.z;
        }
} 
 

// recombineChannels:
// Kernel that combines three given R,G,and B pixel value arrays into
// a single uchar4 vector array.
__global__ 
void recombineChannels(unsigned char *d_r, unsigned char *d_g, unsigned char *d_b, uchar4 *d_orgba, const int num_rows, const int num_cols){
        // Determine location of target pixel in global memory
        int gl_row = blockIdx.y * blockDim.y + threadIdx.y;
        int gl_col = blockIdx.x * blockDim.x + threadIdx.x;

        // Ensure the target pixel is valid
        if (gl_col < num_cols && gl_row < num_rows){
                // Get pixel location
                int pixel_offset = gl_row * num_cols + gl_col;
                // Create uchar4 using three arrays
                d_orgba[pixel_offset] = make_uchar4(d_b[pixel_offset],d_g[pixel_offset],d_r[pixel_offset],255);
        }
} 


void gaussianBlurKernel(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t num_rows, size_t num_cols, 
        unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
        unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
        float *d_filter,  int filterWidth){
 
        // Set grid and block dimensions
        dim3 grid(std::ceil((float)num_cols/(float)BLOCK),std::ceil((float)num_rows/(float)BLOCK),1);
        dim3 block(BLOCK, BLOCK, 1);

        // Seperate out each channel into seperate arrays
        separateChannels<<<grid, block>>>(d_imrgba, d_red, d_green, d_blue, num_rows, num_cols);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        // Compute Gaussian Blur for the red pixel array
        gaussianBlurShared<<<grid, block>>>(d_red, d_rblurred, num_rows, num_cols, d_filter, filterWidth);
        //gaussianBlurGlobal<<<grid, block>>>(d_red, d_rblurred, num_rows, num_cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        // Compute Gaussian Blur for the green pixel array
        gaussianBlurShared<<<grid, block>>>(d_green, d_gblurred, num_rows, num_cols, d_filter, filterWidth);
        //gaussianBlurGlobal<<<grid, block>>>(d_green, d_gblurred, num_rows, num_cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        // Compute Gaussian Blur for the blue pixel array
        gaussianBlurShared<<<grid, block>>>(d_blue, d_bblurred, num_rows, num_cols, d_filter, filterWidth);
        //gaussianBlurGlobal<<<grid, block>>>(d_blue, d_bblurred, num_rows, num_cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        // Recombine the blurred channels into a single uchar4 array
        recombineChannels<<<grid, block>>>(d_rblurred, d_gblurred, d_bblurred, d_oimrgba, num_rows, num_cols);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());   
}




