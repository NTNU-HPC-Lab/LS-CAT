#include "includes.h"
__global__ void kRectifyBoundingBox( float* boxes, float* width_offset, float* height_offset, float* flip, int num_images, int patch_width, int patch_height, int num_locs) {

for (int loc_id = blockIdx.x; loc_id < num_locs; loc_id += gridDim.x) {
float *xmin_block = boxes + num_images * loc_id,
*ymin_block = boxes + num_images * (loc_id + num_locs),
*xmax_block = boxes + num_images * (loc_id + num_locs * 2),
*ymax_block = boxes + num_images * (loc_id + num_locs * 3);

for (int image_id = threadIdx.x; image_id < num_images; image_id += blockDim.x) {
float xmin = (flip[image_id] > 0.5) ? (256.0/patch_width - xmax_block[image_id]) : xmin_block[image_id],
xmax = (flip[image_id] > 0.5) ? (256.0/patch_width - xmin_block[image_id]) : xmax_block[image_id],
ymin = ymin_block[image_id],
ymax = ymax_block[image_id],
wo = width_offset[image_id],
ho = height_offset[image_id];

xmin_block[image_id] = xmin - wo / patch_width;
xmax_block[image_id] = xmax - wo / patch_width;

ymin_block[image_id] = ymin - ho / patch_height;
ymax_block[image_id] = ymax - ho / patch_height;
}
}
}