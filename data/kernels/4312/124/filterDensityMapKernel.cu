#include "includes.h"
__global__ void filterDensityMapKernel( cudaTextureObject_t density_map, unsigned rows, unsigned cols, cudaSurfaceObject_t filter_density_map ) {
const auto x = threadIdx.x + blockIdx.x * blockDim.x;
const auto y = threadIdx.y + blockIdx.y * blockDim.y;
if(x >= cols || y >= rows) return;

const auto half_width = 5;
const float center_density = tex2D<float>(density_map, x, y);

//The window search
float sum_all = 0.0f; float sum_weight = 0.0f;
for(auto y_idx = y - half_width; y_idx <= y + half_width; y_idx++) {
for(auto x_idx = x - half_width; x_idx <= x + half_width; x_idx++) {
const float density = tex2D<float>(density_map, x_idx, y_idx);
const float value_diff2 = (center_density - density) * (center_density - density);
const float pixel_diff2 = (x_idx - x) * (x_idx - x) + (y_idx - y) * (y_idx - y);
const float this_weight = (density > 0.0f) * expf(-(1.0f / 25) * pixel_diff2) * expf(-(1.0f / 0.01) * value_diff2);
sum_weight += this_weight;
sum_all += this_weight * density;
}
}

//The filter value
float filter_density_value = sum_all / (sum_weight);

//Clip the value to suitable range
if(filter_density_value >= 1.0f) {
filter_density_value = 1.0f;
} else if(filter_density_value >= 0.0f) {
//pass
} else {
filter_density_value = 0.0f;
}
//if(isnan(filter_density_value)) printf("Nan in the image");
surf2Dwrite(filter_density_value, filter_density_map, x * sizeof(float), y);
}