#include "includes.h"
__global__ void markPotentialMatchedDepthPairKernel( cudaTextureObject_t index_map, unsigned img_rows, unsigned img_cols, unsigned* reference_pixel_matched_indicator ) {
const auto x = threadIdx.x + blockDim.x*blockIdx.x;
const auto y = threadIdx.y + blockDim.y*blockIdx.y;
if (x >= img_cols || y >= img_rows) return;

//The indicator will must be written to pixel_occupied_array
const auto offset = y * img_cols + x;

//Read the value on index map
const auto surfel_index = tex2D<unsigned>(index_map, x, y);

//Need other criterion?
unsigned indicator = 0;
if(surfel_index != d_invalid_index) {
indicator = 1;
}

reference_pixel_matched_indicator[offset] = indicator;
}