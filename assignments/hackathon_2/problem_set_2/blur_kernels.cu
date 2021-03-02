#include "./gaussian_kernel.h" 

#define O_TILE_WIDTH 4
#define BLOCK_WIDTH (O_TILE_WIDTH + (FILTER_WIDTH - 1))

#define BLOCK 32

///////////////////////////
/// Override Functions ////
///////////////////////////

// atomicAdd:
// Atomic Add function enabling the use of unsigned char values.
/**
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
**/

///////////////////////////
///// Blur Functions //////
///////////////////////////

// DakotaGaussianBlurGlobal:
// Kernel that computes a gaussian blur over a single RGB channel. 
// This implementation uses global memory
__global__ 
void DakotaGaussianBlurGlobal(unsigned char *d_in, unsigned char *d_out, const int num_rows, const int num_cols, float *d_filter, const int filterWidth){
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

// DakotaGaussianBlurShared:
// Kernel taht computes a gaussian blur over a single RGB channel.
// This implementation in specific uses shared memory.
__global__ 
void DakotaGaussianBlurShared(unsigned char *d_in, unsigned char *d_out, const int num_rows, const int num_cols, float *d_filter, const int filterWidth){
        // Given the filter width, determine the correct size of shared memory
        int blur_offset = ((filterWidth-1)/2);
        
        // Create shared memory input array
		extern __shared__ unsigned char input_pixels[];
		// Gather shared memory size
		size_t shared_memory_width = BLOCK + (filterWidth - 1);

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
                int shared_offset = off_sh_row * shared_memory_width + off_sh_col;
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
                                        shared_offset = cur_sh_row * shared_memory_width + off_sh_col;
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
                                        shared_offset = cur_sh_row * shared_memory_width + off_sh_col;
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
                                        shared_offset = off_sh_row * shared_memory_width + cur_sh_col;
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
                                        shared_offset = off_sh_row * shared_memory_width + cur_sh_col;
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
                                                shared_offset = cur_sh_row * shared_memory_width + cur_sh_col;
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
                                                shared_offset = cur_sh_row * shared_memory_width + cur_sh_col;
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
                                                shared_offset = cur_sh_row * shared_memory_width + cur_sh_col;
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
                                                shared_offset = cur_sh_row * shared_memory_width + cur_sh_col;
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
                                        shared_offset = in_sh_row * shared_memory_width + in_sh_col;

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

// DakotaGaussianBlurSharedSepRow:
// Kernel that computes a gaussian blur over a single RGB channel 
// but does this process uses shared memory and splits computations
// by each row of the image and Filter.
__global__ 
void DakotaGaussianBlurSharedSepRow(unsigned char *d_in, float *d_out, const int num_rows, const int num_cols, float *d_filter, const int filterWidth){
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

        // Ensure starting pixel location is valid 
        if (col_offset < num_cols){
                // Load shared memory
                for (int i = 0; i < pixels_per_thread; i++){
                        int pixel_col = col_offset + i;

                        if (pixel_col < num_cols){
                                int global_offset = gl_row * num_cols + pixel_col;
                                input_pixel_row[pixel_col] = d_in[global_offset];
                        }
                }
        }

        // Make sure all threads have loaded before starting computation
        __syncthreads();

        // Ensure starting pixel location is valid 
        if (col_offset < num_cols){
                // Using shared memory, work over pixels per thread
                for (int i = 0; i < pixels_per_thread; i++){
                        // Determine target pixels location
                        int pixel_col = col_offset + i;

                        // Setup loop variables
                        int in_col;
                        float blur_sum = 0;
                        int filter_pos = filter_row * filterWidth;

                        if (pixel_col < num_cols){
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
                                if (result_row >= 0 && result_row < num_rows){
                                        int result_offset = result_row * num_cols + pixel_col;
                                        atomicAdd(d_out + (result_offset), blur_sum);
                                }       
                        }
                }
        }
} 

// DakotaGaussianBlurSharedSepCol:
// Kernel that computes a gaussian blur over a single RGB channel 
// but does this process uses shared memory and splits computations
// by each column of the image and Filter.
__global__ 
void DakotaGaussianBlurSharedSepCol(unsigned char *d_in, float *d_out, const int num_rows, const int num_cols, float *d_filter, const int filterWidth){
        // Create shared memory to hold the full col of values
        extern __shared__ unsigned char input_pixel_col[];

        // Given the filter width, determine the correct col offsets
        int blur_offset = ((filterWidth-1)/2);
        
        // Determine the col this block is working on
        int gl_col = blockIdx.x;
        // Determine the filter col this block is working on
        int filter_col = blockIdx.y;
        // Determine thread id
        int thread_id = threadIdx.x;
        // Determine the number of threads working in each block
        int total_threads = blockDim.x;

        // Determine how many pixels of this col each thread should do
        int pixels_per_thread = std::ceil((float)num_rows/(float)total_threads);

        // Determine the target pixel for each thread by row offset
        int row_offset = thread_id * pixels_per_thread;

        // Ensure starting pixel location is valid 
        if (row_offset < num_rows){
                // Load shared memory
                for (int i = 0; i < pixels_per_thread; i++){
                        int pixel_row = row_offset + i;

                        if (pixel_row < num_rows){
                                int global_offset = pixel_row * num_cols + gl_col;
                                input_pixel_col[pixel_row] = d_in[global_offset];
                        }
                }
        }

        // Make sure all threads have loaded before starting computation
        __syncthreads();

        // Ensure starting pixel location is valid 
        if (row_offset < num_rows){
                // Using shared memory, work over pixels per thread
                for (int i = 0; i < pixels_per_thread; i++){
                        // Determine target pixels location
                        int pixel_row = row_offset + i;

                        // Setup loop variables
                        int in_row;
                        float blur_sum = 0;
                        int filter_pos = filter_col;

                        if (pixel_row < num_rows){
                                // Iterate from the furthest back row to the furthest forward row around target pixel
                                for (in_row = pixel_row - blur_offset; in_row <= pixel_row + blur_offset; in_row++){
                                        // Ensure target blur pixel location is valid
                                        if (in_row >= 0 && in_row < num_rows){
                                                // Multiply current filter location by target pixel and add to running sum
                                                blur_sum += (float)input_pixel_col[in_row] * d_filter[filter_pos];
                                        }
                                        // Always increment filter location
                                        filter_pos = filter_pos + filterWidth;
                                }

                                // Given the current working col, filter col, and blur_offset determine the correct result location
                                int result_col = gl_col + (blur_offset - filter_col);

                                // Store the sum in the correct location of the global results using an atomic Add
                                if (result_col >= 0 && result_col < num_cols){
                                        int result_offset = pixel_row * num_cols + result_col;
                                        atomicAdd(d_out + (result_offset), blur_sum);
                                }       
                        }
                }
        }
} 

// MauriceTileGaussBlurShared:
// Kernel that computes a gaussian blur over a single RGB channel.
// This implementation in specific uses shared memory.
__global__
void MauriceTileGaussBlurShared(unsigned char *d_in, unsigned char *d_out, const int rows, const int cols, float *d_filter, const int filterWidth) {
    
    __shared__ unsigned char tile[BLOCK_WIDTH * BLOCK_WIDTH];

    int col = blockIdx.x * O_TILE_WIDTH + threadIdx.x;
    int row = blockIdx.y * O_TILE_WIDTH + threadIdx.y;
    int i_col = col - (filterWidth / 2);
    int i_row = row - (filterWidth / 2);

    if(threadIdx.y < BLOCK_WIDTH && threadIdx.x < BLOCK_WIDTH) {
        if(i_col >= 0 && i_col < cols && i_row >= 0 && i_row < rows)
            tile[threadIdx.y * BLOCK_WIDTH + threadIdx.x] = d_in[i_row * cols + i_col];
        else
            tile[threadIdx.y * BLOCK_WIDTH + threadIdx.x] = 0;
    }

    __syncthreads();

    if(threadIdx.x < O_TILE_WIDTH && threadIdx.y < O_TILE_WIDTH) {
        if(col < cols && row < rows) {
            float sum = 0;
            int start_row = threadIdx.y;
            int start_col = threadIdx.x;

            for(int i = 0; i < filterWidth; i++) {
                for(int j = 0; j < filterWidth; j++) {
                    int curRow = start_row + i;
                    int curCol = start_col + j;
                    if(curRow >= 0 && curRow < BLOCK_WIDTH && curCol >= 0 && curCol < BLOCK_WIDTH) {
                        sum += (d_filter[i * filterWidth + j] * tile[curRow * BLOCK_WIDTH + curCol]);
                    }
                }
            }

            d_out[row * cols + col] = (unsigned char)sum;
        }
	}
}

// MauriceGaussBlurGlobalSepRow:
// Kernel that computes a gaussian blur over a single RGB channel 
// but does this process uses global memory and splits computations
// by each row of the image and Filter.
__global__
void MauriceGaussBlurGlobalSepRow(unsigned char *d_in, float *d_out, const int rows, const int cols, float *d_filter, const int filterWidth) {
	float sum = 0;
	int filter_offset = filterWidth / 2;
	
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int filter_row = threadIdx.z;
	int acc_col, acc_row;

	if(row < rows && col < cols) {
		for(int i = 0; i < filterWidth; i++) {
			acc_col = col - filter_offset + i;
			if(acc_col >= 0 && acc_col < cols)
				sum += (d_filter[filter_row * filterWidth + i] * d_in[row * cols + acc_col]);
		}

		acc_row = row + (filter_offset - filter_row);
		if (acc_row >= 0 && acc_row < rows)
			atomicAdd(d_out + (acc_row * cols + col), sum);
	}
}

// MauriceGaussBlurGlobalSepCol:
// Kernel that computes a gaussian blur over a single RGB channel 
// but does this process uses global memory and splits computations
// by each col of the image and Filter.
__global__
void MauriceGaussBlurGlobalSepCol(unsigned char *d_in, float *d_out, const int rows, const int cols, float *d_filter, const int filterWidth) {
	float sum = 0;
	int filter_offset = filterWidth / 2;
	
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int filter_col = threadIdx.z;
	int acc_col, acc_row;

	if(row < rows && col < cols) {
		for(int i = 0; i < filterWidth; i++) {
			acc_row = row - filter_offset + i;
			if(acc_row >= 0 && acc_row < rows)
				sum += (d_filter[i * filterWidth + filter_col] * d_in[acc_row * cols + col]);
		}

		acc_col = col + (filter_offset - filter_col);
		if (acc_col >= 0 && acc_col < cols)
			atomicAdd(d_out + (row * cols + acc_col), sum);
	}
}




///////////////////////////
///// Helper Functions ////
///////////////////////////

// DakotaGaussianBlurSepConverter:
// Used to convert seperable results from float to uchar.
// Helps avoid rounding issues.
__global__ 
void DakotaGaussianBlurSepConverter(float *d_in, unsigned char *d_out, const int num_rows, const int num_cols){
        // Determine location of target pixel in global memory
        int gl_row = blockIdx.y * blockDim.y + threadIdx.y;
        int gl_col = blockIdx.x * blockDim.x + threadIdx.x;

        // Ensure the target pixel is valid
        if (gl_col < num_cols && gl_row < num_rows){
                // Get pixel location
                int pixel_offset = gl_row * num_cols + gl_col;

                // Convert and store temp result in the correct global output
                d_out[pixel_offset] = (unsigned char)d_in[pixel_offset];
                // Reset value to zero
                d_in[pixel_offset] = 0.0;
        }
}

// separateChannels:
// Kernel that splits an RGBA uchar4 array into 3 seperate unsigned char 
// arrays.
__global__ 
void separateChannels(uchar4 *d_imrgba, unsigned char *d_r, unsigned char *d_g, unsigned char *d_b, const int rows, const int cols){
	int local_row = blockIdx.y * blockDim.y + threadIdx.y;
	int local_col = blockIdx.x * blockDim.x + threadIdx.x;
	if (local_row >= 0 && local_row < rows && local_col >= 0 && local_col < cols) {
		int index = local_row * cols + local_col;
		uchar4 pixel = d_imrgba[index];
		d_r[index] = pixel.x;
		d_g[index] = pixel.y;
		d_b[index] = pixel.z;
	}
} 
 

// recombineChannels:
// Kernel that combines three given R,G,and B pixel value arrays into
// a single uchar4 vector array.
__global__ 
void recombineChannels(unsigned char *d_r, unsigned char *d_g, unsigned char *d_b, uchar4 *d_orgba, const int rows, const int cols){
	int local_row = blockIdx.y * blockDim.y + threadIdx.y;
	int local_col = blockIdx.x * blockDim.x + threadIdx.x;
	if (local_row >= 0 && local_row < rows && local_col >= 0 && local_col < cols) {
		int index = local_row * cols + local_col;
		d_orgba[index] = make_uchar4(d_b[index], d_g[index], d_r[index], 255);
	}
} 





///////////////////////////
///// Running Kernels /////
///////////////////////////


void MauriceTileGaussBlurKernelShared(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t rows, size_t cols, 
	                    unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
	                    unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
                    	float *d_filter,  int filterWidth){
    int blocks_x = (cols / O_TILE_WIDTH) + ((cols % O_TILE_WIDTH > 0) ? 1 : 0);
    int blocks_y = (rows / O_TILE_WIDTH) + ((rows % O_TILE_WIDTH > 0) ? 1 : 0);

	dim3 tileBlockSize(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    dim3 tileGridSize((cols - 1) / O_TILE_WIDTH + 1, (rows - 1) / O_TILE_WIDTH + 1, 1);

    dim3 blockSize(O_TILE_WIDTH, O_TILE_WIDTH, 1);
    dim3 gridSize(blocks_x, blocks_y, 1);

	separateChannels<<<gridSize, blockSize>>>(d_imrgba, d_red, d_green, d_blue, rows, cols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	MauriceTileGaussBlurShared<<<tileGridSize, tileBlockSize>>>(d_red, d_rblurred, rows, cols, d_filter, filterWidth);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	MauriceTileGaussBlurShared<<<tileGridSize, tileBlockSize>>>(d_green, d_gblurred, rows, cols, d_filter, filterWidth);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	MauriceTileGaussBlurShared<<<tileGridSize, tileBlockSize>>>(d_blue, d_bblurred, rows, cols, d_filter, filterWidth);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	recombineChannels<<<gridSize, blockSize>>>(d_rblurred, d_gblurred, d_bblurred, d_oimrgba, rows, cols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
}

void MauriceGaussBlurKernelGlobalSepRow(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t rows, size_t cols, 
	                    unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
	                    unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
                    	float *d_filter,  int filterWidth, float *tmp_pixels){
    int blocks_x = (cols / O_TILE_WIDTH) + ((cols % O_TILE_WIDTH > 0) ? 1 : 0);
    int blocks_y = (rows / O_TILE_WIDTH) + ((rows % O_TILE_WIDTH > 0) ? 1 : 0);

    dim3 blockSize(O_TILE_WIDTH, O_TILE_WIDTH, filterWidth);
    dim3 gridSize(blocks_x, blocks_y, 1);

	// Set grid and block dimensions for converters
	dim3 grid(std::ceil((float)cols/(float)BLOCK),std::ceil((float)rows/(float)BLOCK),1);
	dim3 block(BLOCK, BLOCK, 1);

	separateChannels<<<gridSize, blockSize>>>(d_imrgba, d_red, d_green, d_blue, rows, cols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	MauriceGaussBlurGlobalSepRow<<<gridSize, blockSize>>>(d_red, tmp_pixels, rows, cols, d_filter, filterWidth);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Convert red pixel results to unsigned chars and reset temp array
	DakotaGaussianBlurSepConverter<<<grid, block>>>(tmp_pixels, d_rblurred, rows, cols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	MauriceGaussBlurGlobalSepRow<<<gridSize, blockSize>>>(d_green, tmp_pixels, rows, cols, d_filter, filterWidth);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Convert green pixel results to unsigned chars and reset temp array
	DakotaGaussianBlurSepConverter<<<grid, block>>>(tmp_pixels, d_gblurred, rows, cols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	MauriceGaussBlurGlobalSepRow<<<gridSize, blockSize>>>(d_blue, tmp_pixels, rows, cols, d_filter, filterWidth);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Convert blue pixel results to unsigned chars and reset temp array
	DakotaGaussianBlurSepConverter<<<grid, block>>>(tmp_pixels, d_bblurred, rows, cols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	recombineChannels<<<gridSize, blockSize>>>(d_rblurred, d_gblurred, d_bblurred, d_oimrgba, rows, cols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
}

void MauriceGaussBlurKernelGlobalSepCol(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t rows, size_t cols, 
	                    unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
	                    unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
                    	float *d_filter,  int filterWidth, float *tmp_pixels){
    int blocks_x = (cols / O_TILE_WIDTH) + ((cols % O_TILE_WIDTH > 0) ? 1 : 0);
    int blocks_y = (rows / O_TILE_WIDTH) + ((rows % O_TILE_WIDTH > 0) ? 1 : 0);

    dim3 blockSize(O_TILE_WIDTH, O_TILE_WIDTH, filterWidth);
    dim3 gridSize(blocks_x, blocks_y, 1);

	// Set grid and block dimensions for converters
	dim3 grid(std::ceil((float)cols/(float)BLOCK),std::ceil((float)rows/(float)BLOCK),1);
	dim3 block(BLOCK, BLOCK, 1);

	separateChannels<<<gridSize, blockSize>>>(d_imrgba, d_red, d_green, d_blue, rows, cols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	MauriceGaussBlurGlobalSepCol<<<gridSize, blockSize>>>(d_red, tmp_pixels, rows, cols, d_filter, filterWidth);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Convert red pixel results to unsigned chars and reset temp array
	DakotaGaussianBlurSepConverter<<<grid, block>>>(tmp_pixels, d_rblurred, rows, cols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	MauriceGaussBlurGlobalSepCol<<<gridSize, blockSize>>>(d_green, tmp_pixels, rows, cols, d_filter, filterWidth);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Convert green pixel results to unsigned chars and reset temp array
	DakotaGaussianBlurSepConverter<<<grid, block>>>(tmp_pixels, d_gblurred, rows, cols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	MauriceGaussBlurGlobalSepCol<<<gridSize, blockSize>>>(d_blue, tmp_pixels, rows, cols, d_filter, filterWidth);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Convert blue pixel results to unsigned chars and reset temp array
	DakotaGaussianBlurSepConverter<<<grid, block>>>(tmp_pixels, d_bblurred, rows, cols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	recombineChannels<<<gridSize, blockSize>>>(d_rblurred, d_gblurred, d_bblurred, d_oimrgba, rows, cols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
}

void DakotaGaussianBlurKernelGlobal(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t num_rows, size_t num_cols, 
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
	DakotaGaussianBlurGlobal<<<grid, block>>>(d_red, d_rblurred, num_rows, num_cols, d_filter, filterWidth);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Compute Gaussian Blur for the green pixel array
	DakotaGaussianBlurGlobal<<<grid, block>>>(d_green, d_gblurred, num_rows, num_cols, d_filter, filterWidth);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Compute Gaussian Blur for the blue pixel array
	DakotaGaussianBlurGlobal<<<grid, block>>>(d_blue, d_bblurred, num_rows, num_cols, d_filter, filterWidth);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Recombine the blurred channels into a single uchar4 array
	recombineChannels<<<grid, block>>>(d_rblurred, d_gblurred, d_bblurred, d_oimrgba, num_rows, num_cols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());   
}

void DakotaGaussianBlurKernelShared(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t num_rows, size_t num_cols, 
	unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
	unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
	float *d_filter,  int filterWidth){

	// Set grid and block dimensions
	// For Global and Shared Memory Format
	dim3 grid(std::ceil((float)num_cols/(float)BLOCK),std::ceil((float)num_rows/(float)BLOCK),1);
	dim3 block(BLOCK, BLOCK, 1);

	// Determine amount of shared memory needed to hold full rows for each kernel call 
	size_t shared_memory_size = BLOCK + (filterWidth - 1);
	shared_memory_size = shared_memory_size * shared_memory_size; 

	// Seperate out each channel into seperate arrays
	separateChannels<<<grid, block>>>(d_imrgba, d_red, d_green, d_blue, num_rows, num_cols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Compute Gaussian Blur for the red pixel array
	DakotaGaussianBlurShared<<<grid, block, shared_memory_size>>>(d_red, d_rblurred, num_rows, num_cols, d_filter, filterWidth);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Compute Gaussian Blur for the green pixel array
	DakotaGaussianBlurShared<<<grid, block, shared_memory_size>>>(d_green, d_gblurred, num_rows, num_cols, d_filter, filterWidth);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Compute Gaussian Blur for the blue pixel array
	DakotaGaussianBlurShared<<<grid, block, shared_memory_size>>>(d_blue, d_bblurred, num_rows, num_cols, d_filter, filterWidth);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Recombine the blurred channels into a single uchar4 array
	recombineChannels<<<grid, block>>>(d_rblurred, d_gblurred, d_bblurred, d_oimrgba, num_rows, num_cols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());   
}

void DakotaGaussianBlurKernelSharedSepRow(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t num_rows, size_t num_cols, 
	unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
	unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
	float *d_filter,  int filterWidth, float *tmp_pixels){

	// Set grid and block dimensions for seperating and recombining
	dim3 grid(std::ceil((float)num_cols/(float)BLOCK),std::ceil((float)num_rows/(float)BLOCK),1);
	dim3 block(BLOCK, BLOCK, 1);

	// Set grid and block dimensions for seperable row gaussian kernel
	dim3 gridSep(num_rows,filterWidth,1);
	dim3 blockSep(BLOCK*BLOCK, 1, 1);

	// Determine amount of shared memory needed to hold full rows for each kernel call 
	size_t shared_memory_size = num_cols * sizeof(unsigned char);

	// Seperate out each channel into seperate arrays
	separateChannels<<<grid, block>>>(d_imrgba, d_red, d_green, d_blue, num_rows, num_cols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Compute Gaussian Blur for the red pixel array
	DakotaGaussianBlurSharedSepRow<<<gridSep, blockSep, shared_memory_size>>>(d_red, tmp_pixels, num_rows, num_cols, d_filter, filterWidth);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Convert red pixel results to unsigned chars and reset temp array
	DakotaGaussianBlurSepConverter<<<grid, block>>>(tmp_pixels, d_rblurred, num_rows, num_cols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Compute Gaussian Blur for the green pixel array
	DakotaGaussianBlurSharedSepRow<<<gridSep, blockSep, shared_memory_size>>>(d_green, tmp_pixels, num_rows, num_cols, d_filter, filterWidth);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Convert green pixel results to unsigned chars and reset temp array
	DakotaGaussianBlurSepConverter<<<grid, block>>>(tmp_pixels, d_gblurred, num_rows, num_cols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Compute Gaussian Blur for the blue pixel array
	DakotaGaussianBlurSharedSepRow<<<gridSep, blockSep, shared_memory_size>>>(d_blue, tmp_pixels, num_rows, num_cols, d_filter, filterWidth);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Convert blue pixel results to unsigned chars and reset temp array
	DakotaGaussianBlurSepConverter<<<grid, block>>>(tmp_pixels, d_bblurred, num_rows, num_cols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Recombine the blurred channels into a single uchar4 array
	recombineChannels<<<grid, block>>>(d_rblurred, d_gblurred, d_bblurred, d_oimrgba, num_rows, num_cols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());   
}

void DakotaGaussianBlurKernelSharedSepCol(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t num_rows, size_t num_cols, 
	unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
	unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
	float *d_filter,  int filterWidth, float *tmp_pixels){

	// Set grid and block dimensions for seperating and recombining
	dim3 grid(std::ceil((float)num_cols/(float)BLOCK),std::ceil((float)num_rows/(float)BLOCK),1);
	dim3 block(BLOCK, BLOCK, 1);

	// Set grid and block dimensions for seperable row gaussian kernel
	dim3 gridSep(num_cols,filterWidth,1);
	dim3 blockSep(BLOCK*BLOCK, 1, 1);

	// Determine amount of shared memory needed to hold full rows for each kernel call 
	size_t shared_memory_size = num_rows * sizeof(unsigned char);

	// Seperate out each channel into seperate arrays
	separateChannels<<<grid, block>>>(d_imrgba, d_red, d_green, d_blue, num_rows, num_cols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Compute Gaussian Blur for the red pixel array
	DakotaGaussianBlurSharedSepCol<<<gridSep, blockSep, shared_memory_size>>>(d_red, tmp_pixels, num_rows, num_cols, d_filter, filterWidth);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Convert red pixel results to unsigned chars and reset temp array
	DakotaGaussianBlurSepConverter<<<grid, block>>>(tmp_pixels, d_rblurred, num_rows, num_cols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Compute Gaussian Blur for the green pixel array
	DakotaGaussianBlurSharedSepCol<<<gridSep, blockSep, shared_memory_size>>>(d_green, tmp_pixels, num_rows, num_cols, d_filter, filterWidth);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Convert green pixel results to unsigned chars and reset temp array
	DakotaGaussianBlurSepConverter<<<grid, block>>>(tmp_pixels, d_gblurred, num_rows, num_cols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Compute Gaussian Blur for the blue pixel array
	DakotaGaussianBlurSharedSepCol<<<gridSep, blockSep, shared_memory_size>>>(d_blue, tmp_pixels, num_rows, num_cols, d_filter, filterWidth);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Convert blue pixel results to unsigned chars and reset temp array
	DakotaGaussianBlurSepConverter<<<grid, block>>>(tmp_pixels, d_bblurred, num_rows, num_cols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Recombine the blurred channels into a single uchar4 array
	recombineChannels<<<grid, block>>>(d_rblurred, d_gblurred, d_bblurred, d_oimrgba, num_rows, num_cols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());   
}