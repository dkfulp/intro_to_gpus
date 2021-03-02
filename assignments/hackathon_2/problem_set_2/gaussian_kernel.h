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

void MauriceTileGaussBlurKernelShared(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t rows, size_t cols, 
	unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
	unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
        float *d_filter,  int filterWidth);

void MauriceGaussBlurKernelGlobalSepRow(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t rows, size_t cols, 
	unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
	unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
        float *d_filter,  int filterWidth, float *tmp_pixels);

void MauriceGaussBlurKernelGlobalSepCol(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t rows, size_t cols, 
	unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
	unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
        float *d_filter,  int filterWidth, float *tmp_pixels);

void DakotaGaussianBlurKernelGlobal(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t num_rows, size_t num_cols, 
	unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
	unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
	float *d_filter,  int filterWidth);

void DakotaGaussianBlurKernelShared(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t num_rows, size_t num_cols, 
	unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
	unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
	float *d_filter,  int filterWidth);

void DakotaGaussianBlurKernelSharedSepRow(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t num_rows, size_t num_cols, 
	unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
	unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
	float *d_filter,  int filterWidth, float *tmp_pixels);

void DakotaGaussianBlurKernelSharedSepCol(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t num_rows, size_t num_cols, 
	unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
	unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
	float *d_filter,  int filterWidth, float *tmp_pixels);