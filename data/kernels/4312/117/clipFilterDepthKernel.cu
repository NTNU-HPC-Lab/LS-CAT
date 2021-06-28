#include "includes.h"
__global__ void clipFilterDepthKernel( cudaTextureObject_t raw_depth, const unsigned clip_img_rows, const unsigned clip_img_cols, const unsigned clip_near, const unsigned clip_far, const float sigma_s_inv_square, const float sigma_r_inv_square, cudaSurfaceObject_t filter_depth ) {
//Parallel over the clipped image
const auto x = threadIdx.x + blockDim.x * blockIdx.x;
const auto y = threadIdx.y + blockDim.y * blockIdx.y;
if (y >= clip_img_rows || x >= clip_img_cols) return;

//Compute the center on raw depth
const auto half_width = 5;
const auto raw_x = x + boundary_clip;
const auto raw_y = y + boundary_clip;
const unsigned short center_depth = tex2D<unsigned short>(raw_depth, raw_x, raw_y);

//Iterate over the window
float sum_all = 0.0f; float sum_weight = 0.0f;
for(auto y_idx = raw_y - half_width; y_idx <= raw_y + half_width; y_idx++) {
for(auto x_idx = raw_x - half_width; x_idx <= raw_x + half_width; x_idx++) {
const unsigned short depth = tex2D<unsigned short>(raw_depth, x_idx, y_idx);
const float depth_diff2 = (depth - center_depth) * (depth - center_depth);
const float pixel_diff2 = (x_idx - raw_x) * (x_idx - raw_x) + (y_idx - raw_y) * (y_idx - raw_y);
const float this_weight = (depth > 0) * expf(-sigma_s_inv_square * pixel_diff2) * expf(-sigma_r_inv_square * depth_diff2);
sum_weight += this_weight;
sum_all += this_weight * depth;
}
}

//Put back to the filtered depth
unsigned short filtered_depth_value = __float2uint_rn(sum_all / sum_weight);
if (filtered_depth_value < clip_near || filtered_depth_value > clip_far) filtered_depth_value = 0;
surf2Dwrite(filtered_depth_value, filter_depth, x * sizeof(unsigned short), y);
}