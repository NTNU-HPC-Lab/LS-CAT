#include "includes.h"
__global__ void markValidIndexMapPixelKernel( cudaTextureObject_t index_map, int validity_halfsize, unsigned img_rows, unsigned img_cols, unsigned char* flatten_validity_indicator ) {
const auto x_center = threadIdx.x + blockDim.x * blockIdx.x;
const auto y_center = threadIdx.y + blockDim.y * blockIdx.y;
if(x_center >= img_cols || y_center >= img_rows) return;
const auto offset = x_center + y_center * img_cols;

//Only depend on this pixel
if(validity_halfsize <= 0) {
const auto surfel_index = tex2D<unsigned>(index_map, x_center, y_center);
unsigned char validity = 0;
if(surfel_index != 0xFFFFFFFF) validity = 1;

//Write it and return
flatten_validity_indicator[offset] = validity;
return;
}

//Should perform a window search as the halfsize is at least 1
unsigned char validity = 1;
for(auto y = y_center - validity_halfsize; y <= y_center + validity_halfsize; y++) {
for(auto x = x_center - validity_halfsize; x <= x_center + validity_halfsize; x++) {
if(tex2D<unsigned>(index_map, x, y) == 0xFFFFFFFF) validity = 0;
}
}

//Save it
flatten_validity_indicator[offset] = validity;
}