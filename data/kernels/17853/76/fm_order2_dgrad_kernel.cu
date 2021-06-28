#include "includes.h"
__global__ void fm_order2_dgrad_kernel(const float* in, const float* top_grad, float* dgrad, int batch_size, int slot_num, int emb_vec_size) {
int tid = threadIdx.x;
int bid = blockIdx.x;

if (tid < emb_vec_size && bid < batch_size) {
float emb_sum = 0.0f;
int offset = bid * slot_num * emb_vec_size + tid;

for (int i = 0; i < slot_num; i++) {
int index = offset + i * emb_vec_size;
emb_sum += in[index];
}
float tgrad = top_grad[bid * emb_vec_size + tid];
for (int i = 0; i < slot_num; i++) {
int index = offset + i * emb_vec_size;
dgrad[index] = tgrad * (emb_sum - in[index]);
}
}
}