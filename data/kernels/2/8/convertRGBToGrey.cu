#include "includes.h"



// Constant values on device
// /!\ undefined in host code, just in kernels /!\ __device__
#define MAX_WEIGHT_VALUES 50
#define MIN_DET FLT_EPSILON

__constant__ __device__ int   LK_iteration;
__constant__ __device__ int   LK_patch;
__constant__ __device__ int   LK_points;
__constant__ __device__ int   LK_height;
__constant__ __device__ int   LK_width;
__constant__ __device__ int   LK_pyr_w;
__constant__ __device__ int   LK_pyr_h;
__constant__ __device__ int   LK_pyr_level;
__constant__ __device__ int   LK_width_offset;
__constant__ __device__ char  LK_init_guess;
__constant__ __device__ float LK_scaling;
__constant__ __device__ float LK_threshold;
__constant__ __device__ float LK_Weight[MAX_WEIGHT_VALUES];
__constant__ __device__ int   LK_win_size;

// Texture buffer is used for each image for on-the-fly interpolation
texture <float, 2, cudaReadModeElementType> texRef_pyramid_prev;
texture <float, 2, cudaReadModeElementType> texRef_pyramid_cur;

// Image pyramids -> texture buffers
texture <float, 2, cudaReadModeElementType> gpu_textr_pict_0;   // pictures > texture space
texture <float, 2, cudaReadModeElementType> gpu_textr_pict_1;

texture <float, 2, cudaReadModeElementType> gpu_textr_deriv_x;  // gradients > texture space
texture <float, 2, cudaReadModeElementType> gpu_textr_deriv_y;

// Convert RGB Picture to grey/float

// Convert Grey uchar picture to float

// Downsample picture to build pyramid lower level (naive implementation..)


// Kernel to compute the tracking

// Kernel to compute the tracking
__global__ void convertRGBToGrey(unsigned char *d_in, float *d_out, int N)
{
int idx = blockIdx.x*blockDim.x + threadIdx.x;

if(idx < N)
{
d_out[idx] = d_in[idx*3]*0.1144f
+ d_in[idx*3+1]*0.5867f
+ d_in[idx*3+2]*0.2989f;
}
}