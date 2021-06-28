#include "includes.h"
__global__ void kOutpTraceMultiplyImages(float *expanded_images, float *expanded_derivs, int num_images, int num_channels, int num_modules_batch, int kernel_size){
int color = blockIdx.y;
int module_id = blockIdx.x;

expanded_images += num_images * num_modules_batch * kernel_size * color;
expanded_images += num_images * module_id;
expanded_derivs += num_images * num_modules_batch * color;
expanded_derivs += num_images * module_id;

for (int kpos = 0; kpos < kernel_size; kpos++) {
for (int im = threadIdx.x; im < num_images; im += blockDim.x) {
int image_idx = im + num_images * num_modules_batch * kpos;
int deriv_idx = im;
expanded_images[image_idx] *= expanded_derivs[deriv_idx];
}
__syncthreads();
}

}