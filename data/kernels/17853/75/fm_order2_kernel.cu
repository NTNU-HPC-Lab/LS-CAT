#include "includes.h"
__global__ void fm_order2_kernel(const float* in, float* out, int batch_size, int slot_num, int emb_vec_size) {
int tid = threadIdx.x;
int bid = blockIdx.x;

if (tid < emb_vec_size && bid < batch_size) {
float emb_sum = 0.0f;
float emb_sum_square = 0.0f;
float emb_square_sum = 0.0f;
int offset = bid * slot_num * emb_vec_size + tid;

for (int i = 0; i < slot_num; i++) {
int index = offset + i * emb_vec_size;
float temp = in[index];
emb_sum += temp;
emb_square_sum += temp * temp;
}
emb_sum_square = emb_sum * emb_sum;

out[bid * emb_vec_size + tid] = 0.5f * (emb_sum_square - emb_square_sum);
}
}