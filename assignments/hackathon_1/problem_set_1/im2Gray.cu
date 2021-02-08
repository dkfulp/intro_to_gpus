#include "im2Gray.h"
#include <math.h> 

#define BLOCK 32

/*
 
  Given an input image d_in, perform the grayscale operation 
  using the luminance formula i.e. 
  o[i] = 0.224f*r + 0.587f*g + 0.111*b; 
  
  Your kernel needs to check for boundary conditions 
  and write the output pixels in gray scale format. 

  you may vary the BLOCK parameter.
 
 */
__global__ 
void im2Gray(uchar4 *d_in, unsigned char *d_grey, int numRows, int numCols){
  // Create shared memory uchar4 to hold tile data of size equal to BLOCK
  __shared__ uchar4 pixels[BLOCK][BLOCK];
  // Create shared memory unsigned char array to hold grey outputs
  __shared__ unsigned char grey_pixels[BLOCK][BLOCK];

  // Get location of pixel in global memory
  int gl_row = blockIdx.y * blockDim.y + threadIdx.y;
  int gl_col = blockIdx.x * blockDim.x + threadIdx.x;

  // Get location of pixel in shared memory
  int sh_row = threadIdx.y;
  int sh_col = threadIdx.x;

  // Load shared memory from global memory
  if (gl_col < numCols && gl_row < numRows){
    int global_offset = gl_row * numCols + gl_col;
    pixels[sh_row][sh_col] = d_in[global_offset];
  }

  // Make sure all threads have loaded before starting computation
  __syncthreads();

  // Compute Grey Pixel and store results in shared memory
  if (gl_col < numCols && gl_row < numRows){
    uchar4 rgba_pixel = pixels[sh_row][sh_col];
    grey_pixels[sh_row][sh_col] = (unsigned char)((float)rgba_pixel.x*0.299f + (float)rgba_pixel.y*0.587f + (float)rgba_pixel.z*0.114f);
  }

  // Make sure all threads have finished computation
  __syncthreads();

  // Write all results back to global memory output array
  if (gl_col < numCols && gl_row < numRows){
    int grey_offset = gl_row * numCols + gl_col;
    d_grey[grey_offset] = grey_pixels[sh_row][sh_col];
  }
}




void launch_im2gray(uchar4 *d_in, unsigned char* d_grey, size_t numRows, size_t numCols){
    // Set grid and block dimensions
    dim3 grid(std::ceil((float)numCols/(float)BLOCK),std::ceil((float)numRows/(float)BLOCK),1);
    dim3 block(BLOCK, BLOCK, 1);

    // Call Kernel
    im2Gray<<<grid,block>>>(d_in, d_grey, numRows, numCols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}





