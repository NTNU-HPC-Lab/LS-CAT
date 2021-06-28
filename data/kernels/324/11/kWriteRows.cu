#include "includes.h"
__global__ void kWriteRows(float* data, float* target, int num_images, int num_modules, int num_modules_batch, int module_id_offset, float beta) {
int c = blockIdx.y;
int src_module_id = blockIdx.x;
int dst_module_id = module_id_offset + blockIdx.x;

data += num_images * (src_module_id + c * num_modules_batch);
target += num_images * (dst_module_id + c * num_modules);

for (int im = threadIdx.x; im < num_images; im += blockDim.x) {
target[im] = beta * data[im];
}
}