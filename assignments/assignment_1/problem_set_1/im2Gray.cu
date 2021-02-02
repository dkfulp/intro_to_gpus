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
  // Column Indicator
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  // Row Indicator
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Only do valid pixel locations
  if(x < numCols && y < numRows) {
    // Get one dimension array offset
    int grey_Offset = y * numCols + x;
    // Get corresponding rgba pixel at this location
    uchar4 rgba_pixel = d_in[grey_Offset];
    // Calculate new grey value
    d_grey[grey_Offset] = (unsigned char)((float)rgba_pixel.x*0.299f + (float)rgba_pixel.y*0.587f + (float)rgba_pixel.z*0.114f);
  }
}




void launch_im2gray(uchar4 *d_in, unsigned char* d_grey, size_t numRows, size_t numCols){
    // Ensure there are not over BLOCK number of blocks
    // Configuration 1
    //dim3 grid(numCols,numRows,1); 

    // Configuration 2
    size_t x = std::ceil((float)BLOCK/2);
    size_t y = std::ceil((float)BLOCK/2);
    std::cout << "x: " << x << " y: " << y << std::endl;
    dim3 grid(x,y,1); 

    // Configuration 3
    //dim3 grid(1,numRows,1);

    // Given the number of total blocks, determine the number of threads needed per block
    // Configuration 1
    //dim3 block(1,1,1); 

    // Configuration 2
    size_t x2 = std::ceil((float)numCols/x);
    size_t y2 = std::ceil((float)numRows/x);
    std::cout << "x2: " << x2 << " y2: " << y2 << std::endl;
    dim3 block(64,64,1);  

    // Configuration 3
    //dim3 block(numCols,1,1); 

    // Call Kernel
    im2Gray<<<grid,block>>>(d_in, d_grey, numRows, numCols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}





