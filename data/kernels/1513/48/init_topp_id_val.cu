#include "includes.h"
__global__ void init_topp_id_val(int* topp_id_val_buf, int* topp_offset_buf, const int batch_size, const int vocab_size)
{
int tid = threadIdx.x;
int bid = blockIdx.x;

if(bid == 0)
{
for(int i = tid; i < batch_size + 1; i+= blockDim.x)
{
topp_offset_buf[i] = i * vocab_size;
}
}

while(tid < vocab_size)
{
topp_id_val_buf[bid * vocab_size + tid] = tid;
tid += blockDim.x;
}
}