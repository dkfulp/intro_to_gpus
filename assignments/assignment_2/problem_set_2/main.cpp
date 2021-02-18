#include <iostream>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <string>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono>

#include "utils.h"
#include "gaussian_kernel.h"

/* 
 * Compute if the two images are correctly 
 * computed. The reference image can 
 * either be produced by a software or by 
 * your own serial implementation.
 * */
void checkApproxResults(unsigned char *ref, unsigned char *gpu, size_t numElems, float eps){
    for (int i = 0; i < numElems; i++){
        if (ref[i] - gpu[i] > eps){
            std::cerr << "Error at position " << i << "\n";

            std::cerr << "Reference:: " << std::setprecision(17) << +ref[i] << "\n";
            std::cerr << "GPU:: " << +gpu[i] << "\n";

            exit(1);
        }
    }
}

void checkResult(const std::string &reference_file, const std::string &output_file, float eps){
    cv::Mat ref_img, out_img;

    ref_img = cv::imread(reference_file, -1);
    out_img = cv::imread(output_file, -1);

    unsigned char *refPtr = ref_img.ptr<unsigned char>(0);
    unsigned char *oPtr = out_img.ptr<unsigned char>(0);

    checkApproxResults(refPtr, oPtr, ref_img.rows * ref_img.cols * ref_img.channels(), eps);
    std::cout << "PASSED!\n";
}

void gaussian_blur_filter(float *arr, const int f_sz, const float f_sigma = 0.2){
    float filterSum = 0.f;
    float norm_const = 0.0; // normalization const for the kernel

    for (int r = -f_sz / 2; r <= f_sz / 2; r++){
        for (int c = -f_sz / 2; c <= f_sz / 2; c++){
            float fSum = expf(-(float)(r * r + c * c) / (2 * f_sigma * f_sigma));
            arr[(r + f_sz / 2) * f_sz + (c + f_sz / 2)] = fSum;
            filterSum += fSum;
        }
    }

    norm_const = 1.f / filterSum;

    for (int r = -f_sz / 2; r <= f_sz / 2; ++r){
        for (int c = -f_sz / 2; c <= f_sz / 2; ++c){
            arr[(r + f_sz / 2) * f_sz + (c + f_sz / 2)] *= norm_const;
        }
    }
}

// Serial implementations of kernel functions
void serialGaussianBlur(unsigned char *in, unsigned char *out, const int num_rows, const int num_cols, float *filter, const int filterWidth){
    // Initialize row and column operators 
    int row, col, in_row, in_col;

    // Iterate over each row
    for (row = 0; row < num_rows; row++){
        // Iterate over each col
        for (col = 0; col < num_cols; col++){
            // Given the filter width, determine the correct row and col offset
            int blur_offset = ((filterWidth-1)/2);

            // Setup current loop variables
            int blur_sum = 0;
            int filter_pos = 0;
            //int blur_pixel_count = 0;

            // Iterate from the furthest back row to the furthest forward row
            for (in_row = row - blur_offset; in_row <= row + blur_offset; in_row++){
                // Iterate from the furthest back col to the furthest forward col
                for (in_col = col - blur_offset; in_col <= col + blur_offset; in_col++){
                    // Ensure target blur pixel location is valid
                    if (in_row >= 0 && in_row < num_rows && in_col >= 0 && in_col < num_cols){
                        // Get target blur pixel offset
                        int pixel_offset = in_row * num_cols + in_col;

                        // Multiply current filter location by target pixel and add to sum
                        blur_sum += (int)( (float)in[pixel_offset] * filter[filter_pos] );
                        // Increment number of pixels used in blur average
                        //blur_pixel_count++;
                    }
                    // Always increment filter location
                    filter_pos++;
                }
            }

            // Divide current sum by the number of elements in the filter
            //int blur_result = (int)( (float)blur_sum / (float)blur_pixel_count );

            // Store results in the correct location of the output array
            int result_offset = row * num_cols + col;
            out[result_offset] = (unsigned char)blur_sum;
        }
    }
}

void serialSeparateChannels(uchar4 *imrgba, unsigned char *r, unsigned char *g, unsigned char *b, const int num_rows, const int num_cols){
    int row, col;

    // Iterate over each row
    for (row = 0; row < num_rows; row++){
        // Iterate over each col
        for (col = 0; col < num_cols; col++){
            // Get pixel location
            int offset = row * num_cols + col;
            // Get corresponding rgba pixel at this location
            uchar4 rgba_pixel = imrgba[offset];
            // Save each pixel element to correct array
            r[offset] = rgba_pixel.x;
            g[offset] = rgba_pixel.y;
            b[offset] = rgba_pixel.z;
        }
    }
}

void serialRecombineChannels(unsigned char *r, unsigned char *g, unsigned char *b, uchar4 *orgba, const int num_rows, const int num_cols){
    int row, col;

    // Iterate over each row
    for (row = 0; row < num_rows; row++){
        // Iterate over each col
        for (col = 0; col < num_cols; col++){
            // Get pixel location
            int offset = row * num_cols + col;
            // Set corresponding output pixel values to equal each corresponding input arrays value
            orgba[offset] = make_uchar4(r[offset],g[offset],b[offset],255);
        }
    }
}

int main(int argc, char const *argv[]){
    // Pointers to host and device input and output images
    uchar4 *h_in_img, *h_o_img;
    uchar4 *d_in_img, *d_o_img;
    uchar4 *r_o_img;

    // Pointers to host and device seperated pixel arrays
    unsigned char *h_red, *h_green, *h_blue;
    unsigned char *h_red_blurred, *h_green_blurred, *h_blue_blurred;
    unsigned char *d_red, *d_green, *d_blue;
    unsigned char *d_red_blurred, *d_green_blurred, *d_blue_blurred;

    // Pointers to host and device filters 
    float *h_filter, *d_filter;
    
    // Matrices to hold input and output images
    cv::Mat imrgba, o_img;

    // Gaussian Blur Parameters
    const int fWidth = 9;
    const float fDev = 3;

    // File String Parameters
    std::string infile;
    std::string outfile;
    std::string reference;

    // Read in input 
    switch (argc){
    case 2:
        infile = std::string(argv[1]);
        outfile = "blurred_gpu.png";
        reference = "blurred_serial.png";
        break;
    case 3:
        infile = std::string(argv[1]);
        outfile = std::string(argv[2]);
        reference = "blurred_serial.png";
        break;
    case 4:
        infile = std::string(argv[1]);
        outfile = std::string(argv[2]);
        reference = std::string(argv[3]);
        break;
    default:
        std::cerr << "Usage ./gblur <in_image> <out_image> <reference_file> \n";
        exit(1);
    }



    // Preprocess
    cv::Mat img = cv::imread(infile.c_str(), cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cerr << "Image file couldn't be read, exiting\n";
        exit(1);
    }
    cv::cvtColor(img, imrgba, cv::COLOR_BGR2RGBA);
    o_img.create(img.rows, img.cols, CV_8UC4);
    const size_t numPixels = img.rows * img.cols;

    // Create pointers to uchar4 arrays 
    h_in_img = (uchar4 *)imrgba.ptr<unsigned char>(0); // pointer to input image
    h_o_img = (uchar4 *)imrgba.ptr<unsigned char>(0);  // pointer to output image
    r_o_img = (uchar4 *)imrgba.ptr<unsigned char>(0);  // pointer to reference output image

    // Allocate and Create the Gaussian filter we plan to use
    h_filter = new float[fWidth * fWidth];
    gaussian_blur_filter(h_filter, fWidth, fDev); // create a filter of 9x9 with std_dev = 0.2

    printArray<float>(h_filter, 81); // printUtility.

    

    // GPU Stage
    /**
    // Allocate all nessecary device memory
    checkCudaErrors(cudaMalloc((void **)&d_in_img, sizeof(uchar4) * numPixels));
    checkCudaErrors(cudaMalloc((void **)&d_o_img, sizeof(uchar4) * numPixels));
    checkCudaErrors(cudaMalloc((void **)&d_red, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMalloc((void **)&d_green, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMalloc((void **)&d_blue, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMalloc((void **)&d_red_blurred, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMalloc((void **)&d_green_blurred, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMalloc((void **)&d_blue_blurred, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMalloc((void **)&d_filter, sizeof(float) * fWidth * fWidth));

    // Copy image and filter to device global memory
    checkCudaErrors(cudaMemcpy(d_in_img, h_in_img, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * fWidth * fWidth, cudaMemcpyHostToDevice));

    // Launch Kernel Code
    your_gauss_blur(d_in_img, d_o_img, img.rows, img.cols, d_red, d_green, d_blue,
                    d_red_blurred, d_green_blurred, d_blue_blurred, d_filter, fWidth);

    // Copy the output image from device to host
    checkCudaErrors(cudaMemcpy(h_o_img, d_o_img, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost));
    **/



    // Serial Stage
    // Allocate all nessecary host memory
    h_red = new unsigned char[numPixels];
    h_green = new unsigned char[numPixels];
    h_blue = new unsigned char[numPixels];
    h_red_blurred = new unsigned char[numPixels];
    h_green_blurred = new unsigned char[numPixels];
    h_blue_blurred = new unsigned char[numPixels];

    // Start serial timing
    auto start = std::chrono::high_resolution_clock::now(); 
    
    // Seperate each of the RGB channels
    serialSeparateChannels(h_in_img, h_red, h_green, h_blue, img.rows, img.cols);

    // Perform Gaussian Blur over each pixel channnel
    serialGaussianBlur(h_red, h_red_blurred, img.rows, img.cols, h_filter, fWidth);
    serialGaussianBlur(h_green, h_green_blurred, img.rows, img.cols, h_filter, fWidth);
    serialGaussianBlur(h_blue, h_blue_blurred, img.rows, img.cols, h_filter, fWidth);

    // Recombine each of the RGB channels
    serialRecombineChannels(h_red_blurred, h_green_blurred, h_blue_blurred, r_o_img, img.rows, img.cols);

    // End serial timing
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); 
    std::cout << "Serial Duration: " << duration.count() << " micro seconds" << std::endl; 



    // Output Stage 
    // Create output image using GPU Results
    //cv::Mat output(img.rows, img.cols, CV_8UC4, (void *)h_o_img); // generate GPU output image.
    //bool suc = cv::imwrite(outfile.c_str(), output);
    //if (!suc){
    //    std::cerr << "Couldn't write GPU image!\n";
    //    exit(1);
    //}
    // Create output image using Serial Results
    cv::Mat output_s(img.rows, img.cols, CV_8UC4, (void *)r_o_img); // generate serial output image.
    bool suc = cv::imwrite(reference.c_str(), output_s);
    if (!suc){
        std::cerr << "Couldn't write serial image!\n";
        exit(1);
    }

    // Compare results to ensure accuracy
    //checkResult(reference, outfile, 1e-5);

    // Free Allocated Memory
    //cudaFree(d_in_img);
    //cudaFree(d_o_img);
    //cudaFree(d_red);
    //cudaFree(d_green);
    //cudaFree(d_blue);
    //cudaFree(d_red_blurred);
    //cudaFree(d_green_blurred);
    //cudaFree(d_blue_blurred);
    //cudaFree(d_filter);
    delete[] h_red;
    delete[] h_green;
    delete[] h_blue;
    delete[] h_red_blurred;
    delete[] h_green_blurred;
    delete[] h_blue_blurred;
    delete[] h_filter;
    return 0;
}
