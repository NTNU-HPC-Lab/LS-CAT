#include "includes.h"
__global__ void kExpand(float *images, float* targets, int num_images, int num_input_channels, int image_size_y, int image_size_x, int num_modules_y, int num_modules_x, int kernel_size_y, int kernel_size_x, int padding_y, int padding_x, int stride_y, int stride_x, int num_modules_batch, int module_id_offset) {
int color = blockIdx.y;
int src_module_id = module_id_offset + blockIdx.x;
int dst_module_id = blockIdx.x;

int module_id_x = src_module_id % num_modules_x;
int module_id_y = src_module_id / num_modules_x;
int startX = module_id_x * stride_x + padding_x;
int startY = module_id_y * stride_y + padding_y;
int Y, X;
long target_id, source_id;
images += num_images * image_size_x * image_size_y * color;
targets += num_images * (dst_module_id + num_modules_batch * (kernel_size_y * kernel_size_x * color));
for (int y = 0; y < kernel_size_y; y++) {
Y = startY + y;
for (int x = 0; x < kernel_size_x; x++) {
X = startX + x;
target_id = num_images * num_modules_batch * (x + kernel_size_x * y);
source_id = num_images * (X + image_size_x * Y);
if (X < 0 || X >= image_size_x || Y < 0 || Y >= image_size_y) {
for (int im = threadIdx.x; im < num_images; im += blockDim.x) {
targets[target_id + im] = 0;
}
} else {
for (int im = threadIdx.x; im < num_images; im += blockDim.x) {
targets[target_id + im] = images[source_id + im];
}
}
__syncthreads();
}
}
}