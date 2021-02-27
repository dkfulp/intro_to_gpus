#include "./gaussian_kernel.h" 

#define BLOCK 32
#define FILTER_WIDTH 9
// Shared Memory = BLOCK + FILTER_WIDTH - 1
#define SHARED_MEM_SIZE 40



// atomicAdd function
inline __device__ void atomicAdd(unsigned char *address, unsigned char val) {
        size_t offset = (size_t)address & 3;
        uint32_t * address_as_ui = (uint32_t *)((char *)address - offset);
        uint32_t old = *address_as_ui;
        uint32_t shift = offset * 8;
        uint32_t old_byte;
        uint32_t newval;
        uint32_t assumed;
    
        do {
                    assumed = old;
                    old_byte = (old >> shift) & 0xff;
                    // preserve size in initial cast. Casting directly to uint32_t pads
                    // negative signed values with 1's (e.g. signed -1 = unsigned ~0).
                    newval = static_cast<uint8_t>(val + old_byte);
                    newval = (old & ~(0x000000ff << shift)) | (newval << shift);
                    old = atomicCAS(address_as_ui, assumed, newval);
        } while (assumed != old);
}


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
                float blur_sum = 0;
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
                                        blur_sum += (float)d_in[pixel_offset] * d_filter[filter_pos];
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


// gaussianBlurSharedv1:
// Kernel that computes a gaussian blur over a single RGB channel. 
// This implementation in specific uses shared memory to reduce the 
// number of accesses to global memory to improve performance.
 __global__ 
 void gaussianBlurSharedv1(unsigned char *d_in, unsigned char *d_out, const int num_rows, const int num_cols, float *d_filter, const int filterWidth){
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
                float blur_sum = 0;
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
                                        blur_sum += (float)input_pixels[shared_offset] * d_filter[filter_pos];
                                // Target Pixel is Not In Shared Memory
                                } else {
                                        // Ensure target pixel global location is valid
                                        in_gl_row = blockIdx.y * blockDim.y + in_row;
                                        in_gl_col = blockIdx.x * blockDim.x + in_col;
                                        if (in_gl_row >= 0 && in_gl_row < num_rows && in_gl_col >= 0 && in_gl_col < num_cols){
                                                // Get target blur pixel offset
                                                int global_offset = in_gl_row * num_cols + in_gl_col;

                                                // Multiply current filter location by target pixel and add to running sum
                                                blur_sum += (float)d_in[global_offset] * d_filter[filter_pos];
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



// gaussianBlurSharedv2:
// Kernel that computes a gaussian blur over a single RGB channel. 
// This implementation in specific uses shared memory to reduce the 
// number of accesses to global memory to improve performance.
// 
// *** Note: To use this approach, the block size and filter width 
// must be known before hand and set at the top of the document.
__global__ 
void gaussianBlurSharedv2(unsigned char *d_in, unsigned char *d_out, const int num_rows, const int num_cols, float *d_filter){
        // Given the filter width, determine the correct size of shared memory
        int blur_offset = ((FILTER_WIDTH-1)/2);
        
        // Create shared memory input array
        __shared__ unsigned char input_pixels[SHARED_MEM_SIZE * SHARED_MEM_SIZE];

        // Get location of pixel in global memory
        int gl_row = blockIdx.y * blockDim.y + threadIdx.y;
        int gl_col = blockIdx.x * blockDim.x + threadIdx.x;

        // Get location of pixel in true block data
        int tr_sh_row = threadIdx.y;
        int tr_sh_col = threadIdx.x;

        // Get location of pixel in shared memory 
        int off_sh_row = tr_sh_row + blur_offset;
        int off_sh_col = tr_sh_col + blur_offset;

        // Load shared memory with edge pixels loading extra pixels
        // Ensure working pixel is valid
        if (gl_col < num_cols && gl_row < num_rows){
                // Each pixel loads in its own data from global memory
                int global_offset = gl_row * num_cols + gl_col;
                int shared_offset = off_sh_row * SHARED_MEM_SIZE + off_sh_col;
                input_pixels[shared_offset] = d_in[global_offset];

                // Top Row Edge Pixels
                if (tr_sh_row == 0){
                        // Load in pixels above equal to blur_offset
                        for (int i = 1; i <= blur_offset; i++){
                                int cur_gl_row = gl_row - i;
                                int cur_sh_row = off_sh_row - i;
                                // Ensure target global pixel is valid
                                if (cur_gl_row >= 0){
                                        // If valid, save pixel to shared memory
                                        global_offset = cur_gl_row * num_cols + gl_col;
                                        shared_offset = cur_sh_row * SHARED_MEM_SIZE + off_sh_col;
                                        input_pixels[shared_offset] = d_in[global_offset]; 
                                }
                        }
                } 

                // Bottom Row Edge Pixels   
                if (tr_sh_row == blockDim.y-1){
                        // Load in pixels above equal to blur_offset
                        for (int i = 1; i <= blur_offset; i++){
                                int cur_gl_row = gl_row + i;
                                int cur_sh_row = off_sh_row + i;
                                // Ensure target global pixel is valid
                                if (cur_gl_row < num_rows){
                                        // If valid, save pixel to shared memory
                                        global_offset = cur_gl_row * num_cols + gl_col;
                                        shared_offset = cur_sh_row * SHARED_MEM_SIZE + off_sh_col;
                                        input_pixels[shared_offset] = d_in[global_offset]; 
                                }
                        }
                }

                // Left Column Edge Pixels   
                if (tr_sh_col == 0){
                        // Load in pixels above equal to blur_offset
                        for (int i = 1; i <= blur_offset; i++){
                                int cur_gl_col = gl_col - i;
                                int cur_sh_col = off_sh_col - i;
                                // Ensure target global pixel is valid
                                if (cur_gl_col >= 0){
                                        // If valid, save pixel to shared memory
                                        global_offset = gl_row * num_cols + cur_gl_col;
                                        shared_offset = off_sh_row * SHARED_MEM_SIZE + cur_sh_col;
                                        input_pixels[shared_offset] = d_in[global_offset];
                                }
                        } 
                } 

                // Right Column Edge Pixels  
                if (tr_sh_col == blockDim.x-1){
                        // Load in pixels above equal to blur_offset
                        for (int i = 1; i <= blur_offset; i++){
                                int cur_gl_col = gl_col + i;
                                int cur_sh_col = off_sh_col + i;
                                // Ensure target global pixel is valid
                                if (cur_gl_col < num_cols){
                                        // If valid, save pixel to shared memory
                                        global_offset = gl_row * num_cols + cur_gl_col;
                                        shared_offset = off_sh_row * SHARED_MEM_SIZE + cur_sh_col;
                                        input_pixels[shared_offset] = d_in[global_offset];
                                }
                        }
                }

                // Upper Left Corner Pixel
                if (tr_sh_row == 0 && tr_sh_col == 0){
                        // Load in pixels diagonal equal to blur_offset
                        for (int i = 1; i <= blur_offset; i++){
                                for (int j = 1; j <= blur_offset; j++){
                                        int cur_gl_row = gl_row - i;
                                        int cur_gl_col = gl_col - j;
                                        int cur_sh_row = off_sh_row - i;
                                        int cur_sh_col = off_sh_col - j;
                                        // Ensure target global pixel is valid
                                        if (cur_gl_row >= 0 && cur_gl_col >= 0){
                                                // If valid, save pixel to shared memory
                                                global_offset = cur_gl_row * num_cols + cur_gl_col;
                                                shared_offset = cur_sh_row * SHARED_MEM_SIZE + cur_sh_col;
                                                input_pixels[shared_offset] = d_in[global_offset]; 
                                        }

                                }   
                        }
                }

                // Upper Right Corner Pixel
                if (tr_sh_row == 0 && tr_sh_col == blockDim.x-1){
                        // Load in pixels diagonal equal to blur_offset
                        for (int i = 1; i <= blur_offset; i++){
                                for (int j = 1; j <= blur_offset; j++){
                                        int cur_gl_row = gl_row - i;
                                        int cur_gl_col = gl_col + j;
                                        int cur_sh_row = off_sh_row - i;
                                        int cur_sh_col = off_sh_col + j;
                                        // Ensure target global pixel is valid
                                        if (cur_gl_row >= 0 && cur_gl_col < num_cols){
                                                // If valid, save pixel to shared memory
                                                global_offset = cur_gl_row * num_cols + cur_gl_col;
                                                shared_offset = cur_sh_row * SHARED_MEM_SIZE + cur_sh_col;
                                                input_pixels[shared_offset] = d_in[global_offset]; 
                                        }
                                }   
                        }
                }

                // Lower Left Corner Pixel
                if (tr_sh_row == blockDim.y-1 && tr_sh_col == 0){
                        // Load in pixels diagonal equal to blur_offset
                        for (int i = 1; i <= blur_offset; i++){
                                for (int j = 1; j <= blur_offset; j++){
                                        int cur_gl_row = gl_row + i;
                                        int cur_gl_col = gl_col - j;
                                        int cur_sh_row = off_sh_row + i;
                                        int cur_sh_col = off_sh_col - j;
                                        // Ensure target global pixel is valid
                                        if (cur_gl_row < num_rows && cur_gl_col >= 0){
                                                // If valid, save pixel to shared memory
                                                global_offset = cur_gl_row * num_cols + cur_gl_col;
                                                shared_offset = cur_sh_row * SHARED_MEM_SIZE + cur_sh_col;
                                                input_pixels[shared_offset] = d_in[global_offset]; 
                                        }
                                }   
                        }
                }

                // Lower Right Corner Pixel
                if (tr_sh_row == blockDim.y-1 && tr_sh_col == blockDim.x-1){
                        // Load in pixels diagonal equal to blur_offset
                        for (int i = 1; i <= blur_offset; i++){
                                for (int j = 1; j <= blur_offset; j++){
                                        int cur_gl_row = gl_row + i;
                                        int cur_gl_col = gl_col + j;
                                        int cur_sh_row = off_sh_row + i;
                                        int cur_sh_col = off_sh_col + j;
                                        // Ensure target global pixel is valid
                                        if (cur_gl_row < num_rows && cur_gl_col < num_cols){
                                                // If valid, save pixel to shared memory
                                                global_offset = cur_gl_row * num_cols + cur_gl_col;
                                                shared_offset = cur_sh_row * SHARED_MEM_SIZE + cur_sh_col;
                                                input_pixels[shared_offset] = d_in[global_offset]; 
                                        }
                                }   
                        }
                }

                // Make sure all threads have loaded before starting computation
                __syncthreads();

                // Begin Calculations by Setting up Loop Variables
                int row_offset, col_offset;
                int in_sh_row, in_sh_col;
                int in_gl_row, in_gl_col;
                float blur_sum = 0;
                int filter_pos = 0;
              
                // Iterate from the furthest back offset shared row to the furthest forward offset shared row
                for (row_offset = - blur_offset; row_offset <= blur_offset; row_offset++){
                        // Iterate from the furthest back offset shared col to the furthest forward offset shared col
                        for (col_offset = - blur_offset; col_offset <= blur_offset; col_offset++){
                                // Calculate global and shared offsets
                                in_sh_row = off_sh_row + row_offset;
                                in_sh_col = off_sh_col + col_offset;
                                in_gl_row = gl_row + row_offset;
                                in_gl_col = gl_col + col_offset;

                                // Ensure target blur pixel location is valid
                                if (in_gl_row < num_rows && in_gl_col < num_cols && in_gl_row >= 0 && in_gl_col >= 0){
                                        // Get target blur pixel from shared memory
                                        shared_offset = in_sh_row * SHARED_MEM_SIZE + in_sh_col;

                                        // Multiply current filter location by target pixel and add to running sum
                                        blur_sum += (float)input_pixels[shared_offset] * d_filter[filter_pos];

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


// gaussianBlurSepRow:
// Kernel that computes a gaussian blur over a single RGB channel 
// but does this process uses shared memory and splits computations
// by each row of the image and Filter.
__global__ 
void gaussianBlurSepRow(unsigned char *d_in, unsigned char *d_out, const int num_rows, const int num_cols, float *d_filter, const int filterWidth){
        // Create shared memory to hold the full row of values
        extern __shared__ unsigned char input_pixel_row[];

        // Given the filter width, determine the correct col offsets
        int blur_offset = ((filterWidth-1)/2);
        
        // Determine the row this block is working on
        int gl_row = blockIdx.x;
        // Determine the filter row this block is working on
        int filter_row = blockIdx.y;
        // Determine thread id
        int thread_id = threadIdx.x;
        // Determine the number of threads working in each block
        int total_threads = blockDim.x;

        // Determine how many pixels of this row each thread should do
        int pixels_per_thread = std::ceil((float)num_cols/(float)total_threads);

        // Determine the target pixel for each thread by col offset
        int col_offset = thread_id * pixels_per_thread;

        // Load shared memory
        for (int i = 0; i < pixels_per_thread; i++){
                int pixel_col = col_offset + i;

                if (pixel_col < num_cols){
                        int global_offset = gl_row * num_cols + pixel_col;
                        input_pixel_row[pixel_col] = d_in[global_offset];
                }
        }

        // Make sure all threads have loaded before starting computation
        __syncthreads();

        // Setup loop variables
        float blur_sum = 0;
        int in_col;

        // Using shared memory, work over pixels per thread
        for (int i = 0; i < pixels_per_thread; i++){
                // Determine target pixels index
                int pixel_col = col_offset + i;
                // Reset filter position
                int filter_pos = filter_row * filterWidth;

                // Iterate from the furthest back col to the furthest forward col around target pixel
                for (in_col = pixel_col - blur_offset; in_col <= pixel_col + blur_offset; in_col++){
                        // Ensure target blur pixel location is valid
                        if (in_col >= 0 && in_col < num_cols){
                                // Multiply current filter location by target pixel and add to running sum
                                blur_sum += (float)input_pixel_row[in_col] * d_filter[filter_pos];
                        }
                        // Always increment filter location
                        filter_pos++;
                }

                // Given the current working row, filter row, and blur_offset determine the correct result location
                int result_row = gl_row + (blur_offset - filter_row);

                // Store the sum in the correct location of the global results using an atomic Add
                int result_offset = result_row * num_cols + pixel_col;
                atomicAdd(d_out + (result_offset), (unsigned char)__float2uint_rd(blur_sum));
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


void gaussianBlurKernelGlobal(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t num_rows, size_t num_cols, 
        unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
        unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
        float *d_filter,  int filterWidth){
 
        // Set grid and block dimensions
        // For Global and Shared Memory Format
        dim3 grid(std::ceil((float)num_cols/(float)BLOCK),std::ceil((float)num_rows/(float)BLOCK),1);
        dim3 block(BLOCK, BLOCK, 1);

        // Seperate out each channel into seperate arrays
        separateChannels<<<grid, block>>>(d_imrgba, d_red, d_green, d_blue, num_rows, num_cols);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        // Compute Gaussian Blur for the red pixel array
        gaussianBlurGlobal<<<grid, block>>>(d_red, d_rblurred, num_rows, num_cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        // Compute Gaussian Blur for the green pixel array
        gaussianBlurGlobal<<<grid, block>>>(d_green, d_gblurred, num_rows, num_cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        // Compute Gaussian Blur for the blue pixel array
        gaussianBlurGlobal<<<grid, block>>>(d_blue, d_bblurred, num_rows, num_cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        // Recombine the blurred channels into a single uchar4 array
        recombineChannels<<<grid, block>>>(d_rblurred, d_gblurred, d_bblurred, d_oimrgba, num_rows, num_cols);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());   
}

void gaussianBlurKernelSharedv1(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t num_rows, size_t num_cols, 
        unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
        unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
        float *d_filter,  int filterWidth){
 
        // Set grid and block dimensions
        // For Global and Shared Memory Format
        dim3 grid(std::ceil((float)num_cols/(float)BLOCK),std::ceil((float)num_rows/(float)BLOCK),1);
        dim3 block(BLOCK, BLOCK, 1);

        // Seperate out each channel into seperate arrays
        separateChannels<<<grid, block>>>(d_imrgba, d_red, d_green, d_blue, num_rows, num_cols);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        // Compute Gaussian Blur for the red pixel array
        gaussianBlurSharedv1<<<grid, block>>>(d_red, d_rblurred, num_rows, num_cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        // Compute Gaussian Blur for the green pixel array
        gaussianBlurSharedv1<<<grid, block>>>(d_green, d_gblurred, num_rows, num_cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        // Compute Gaussian Blur for the blue pixel array
        gaussianBlurSharedv1<<<grid, block>>>(d_blue, d_bblurred, num_rows, num_cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        // Recombine the blurred channels into a single uchar4 array
        recombineChannels<<<grid, block>>>(d_rblurred, d_gblurred, d_bblurred, d_oimrgba, num_rows, num_cols);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());   
}

void gaussianBlurKernelSharedv2(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t num_rows, size_t num_cols, 
        unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
        unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
        float *d_filter){
 
        // Set grid and block dimensions
        // For Global and Shared Memory Format
        dim3 grid(std::ceil((float)num_cols/(float)BLOCK),std::ceil((float)num_rows/(float)BLOCK),1);
        dim3 block(BLOCK, BLOCK, 1);

        // Seperate out each channel into seperate arrays
        separateChannels<<<grid, block>>>(d_imrgba, d_red, d_green, d_blue, num_rows, num_cols);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        // Compute Gaussian Blur for the red pixel array
        gaussianBlurSharedv2<<<grid, block>>>(d_red, d_rblurred, num_rows, num_cols, d_filter);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        // Compute Gaussian Blur for the green pixel array
        gaussianBlurSharedv2<<<grid, block>>>(d_green, d_gblurred, num_rows, num_cols, d_filter);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        // Compute Gaussian Blur for the blue pixel array
        gaussianBlurSharedv2<<<grid, block>>>(d_blue, d_bblurred, num_rows, num_cols, d_filter);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        // Recombine the blurred channels into a single uchar4 array
        recombineChannels<<<grid, block>>>(d_rblurred, d_gblurred, d_bblurred, d_oimrgba, num_rows, num_cols);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());   
}

void gaussianBlurKernelSharedSepRow(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t num_rows, size_t num_cols, 
        unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
        unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
        float *d_filter,  int filterWidth){
 
        // Set grid and block dimensions for seperating and recombining
        dim3 grid(std::ceil((float)num_cols/(float)BLOCK),std::ceil((float)num_rows/(float)BLOCK),1);
        dim3 block(BLOCK, BLOCK, 1);

        // Set grid and block dimensions for seperable row gaussian kernel
        dim3 gridSep(num_rows,filterWidth,1);
        size_t block_size = BLOCK*BLOCK;
        if (num_cols < block_size){
                block_size = num_cols;
        } 
        dim3 blockSep(block_size, 1, 1);

        // Determine amount of shared memory needed to hold full rows for each kernel call 
        size_t shared_memory_size = num_cols * sizeof(unsigned char);

        // Seperate out each channel into seperate arrays
        separateChannels<<<grid, block>>>(d_imrgba, d_red, d_green, d_blue, num_rows, num_cols);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        // Compute Gaussian Blur for the red pixel array
        gaussianBlurSepRow<<<gridSep, blockSep, shared_memory_size>>>(d_red, d_rblurred, num_rows, num_cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        // Compute Gaussian Blur for the green pixel array
        gaussianBlurSepRow<<<gridSep, blockSep, shared_memory_size>>>(d_green, d_gblurred, num_rows, num_cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        // Compute Gaussian Blur for the blue pixel array
        gaussianBlurSepRow<<<gridSep, blockSep, shared_memory_size>>>(d_blue, d_bblurred, num_rows, num_cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        // Recombine the blurred channels into a single uchar4 array
        recombineChannels<<<grid, block>>>(d_rblurred, d_gblurred, d_bblurred, d_oimrgba, num_rows, num_cols);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());   
}




