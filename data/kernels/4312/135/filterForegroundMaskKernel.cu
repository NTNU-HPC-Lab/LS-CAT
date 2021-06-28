#include "includes.h"
__global__ void filterForegroundMaskKernel( cudaTextureObject_t foreground_mask, unsigned mask_rows, unsigned mask_cols, const float sigma, cudaSurfaceObject_t filter_mask ) {
const int x = threadIdx.x + blockDim.x * blockIdx.x;
const int y = threadIdx.y + blockDim.y * blockIdx.y;
if(x >= mask_cols || y >= mask_rows) return;

//A window search
const int halfsize = __float2uint_ru(sigma) * 2;
float total_weight = 0.0f;
float total_value = 0.0f;
for(int neighbor_y = y - halfsize; neighbor_y <= y + halfsize; neighbor_y++) {
for(int neighbor_x = x - halfsize; neighbor_x <= x + halfsize; neighbor_x++) {
//Retrieve the mask value at neigbour
const unsigned char neighbor_foreground = tex2D<unsigned char>(foreground_mask, neighbor_x, neighbor_y);

//Compute the gaussian weight
const float diff_x_square = (neighbor_x - x) * (neighbor_x - x);
const float diff_y_square = (neighbor_y - y) * (neighbor_y - y);
const float weight = __expf(0.5f * (diff_x_square + diff_y_square) / (sigma * sigma));

//Accumlate it
if(neighbor_x >= 0 && neighbor_x < mask_cols && neighbor_y >= 0 && neighbor_y < mask_rows)
{
total_weight += weight;
total_value += weight * float(1 - neighbor_foreground);
}
}
}


//Compute the value locally
const unsigned char foreground_indicator = tex2D<unsigned char>(foreground_mask, x, y);
float filter_value = 0.0;
if(foreground_indicator == 0) {
filter_value = total_value / (total_weight + 1e-3f);
}


//Write to the surface
surf2Dwrite(filter_value, filter_mask, x * sizeof(float), y);
}