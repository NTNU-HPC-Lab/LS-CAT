#include "includes.h"
__global__ void frame_encode_staircase(int *data, int *pcm,int param_k,int param_m,int w_f,int packet_size)
{
int ps = packet_size/sizeof(int);

int x  = threadIdx.x;

for (int index = param_k; index < param_k + param_m-1; index++)
{

int offset = x;

while (offset < ps)
{
// *((int *)(data + (index+1)*ps + offset + intSize * x)) ^= *((int *)(data + index * ps + intSize * x + offset));
data[(index+1)*ps + offset] ^= data[index*ps + offset];
offset += blockDim.x;
}



}


}