#include "includes.h"
__global__ void kExtractPatches(float* images, float* patches, float* indices, float* width_offset, float* height_offset, int num_images, int img_width, int img_height, int patch_width, int patch_height, int num_colors) {
const unsigned long idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned long numThreads = blockDim.x * gridDim.x;
const unsigned long total_pixels = patch_width * patch_height * num_colors * num_images;
unsigned long ind, pos;
unsigned long image_id, dest_row, dest_col, color, source_row, source_col;
for (unsigned long i = idx; i < total_pixels; i += numThreads) {
ind = i;
image_id = ind % num_images; ind /= num_images;
dest_col = ind % patch_width; ind /= patch_width;
dest_row = ind % patch_height; ind /= patch_height;
color = ind % num_colors;

source_row = int(height_offset[image_id]) + dest_row;
source_col = int(width_offset[image_id]) + dest_col;
pos = img_width * img_height * num_colors * (int)indices[image_id] + img_width * img_height * color + img_width * source_row + source_col;
patches[i] = images[pos];
}
}