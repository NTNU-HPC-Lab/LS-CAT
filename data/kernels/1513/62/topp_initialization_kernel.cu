#include "includes.h"
__global__ void topp_initialization_kernel(bool* finished, int* sequence_length, int* word_ids, int* topp_id_val_buf, int* topp_offset_buf, const int batch_size, const int vocab_size, const int start_id)
{
int tid = threadIdx.x;
int bid = blockIdx.x;

if(bid == 0)
{
for(int i = tid; i < batch_size + 1; i+= blockDim.x)
{
topp_offset_buf[i] = i * vocab_size;
}

for(int i = tid; i < batch_size; i+= blockDim.x)
{
finished[i] = false;
sequence_length[i] = 0;
word_ids[i] = start_id;
}
}

int index = tid + bid * blockDim.x;
while(index < batch_size * vocab_size)
{
topp_id_val_buf[index] = index % vocab_size;
index += blockDim.x * gridDim.x;
}
}