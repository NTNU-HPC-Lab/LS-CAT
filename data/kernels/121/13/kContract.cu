#include "includes.h"
__global__ void kContract(float *expanded_data, float* targets, int num_images, int num_input_channels, int image_size_y, int image_size_x, int num_modules_y, int num_modules_x, int kernel_size_y, int kernel_size_x, int padding_y, int padding_x, int stride_y, int stride_x, int num_modules_batch, int module_id_offset) {
int color = blockIdx.y;
int dst_module_id = module_id_offset + blockIdx.x;
int src_module_id = blockIdx.x;

int module_id_x = dst_module_id % num_modules_x;
int module_id_y = dst_module_id / num_modules_x;
int startX = module_id_x * stride_x + padding_x;
int startY = module_id_y * stride_y + padding_y;
int Y, X;
long target_id, source_id;
targets += num_images * image_size_x * image_size_y * color;
expanded_data  += num_images * (src_module_id + num_modules_batch * (kernel_size_y * kernel_size_x * color));
for (int y = 0; y < kernel_size_y; y++) {
Y = startY + y;
for (int x = 0; x < kernel_size_x; x++) {
X = startX + x;
source_id = num_images * num_modules_batch * (x + kernel_size_x * y);
target_id = num_images * (X + image_size_x * Y);
if (X < 0 || X >= image_size_x || Y < 0 || Y >= image_size_y) {
// do nothing.
} else {
for (int im = threadIdx.x; im < num_images; im += blockDim.x) {
atomicAdd(&targets[target_id + im], expanded_data[source_id + im]);
__syncthreads();
}
}
}
}
}