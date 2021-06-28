#include "includes.h"
__global__ void getMaxPorb(const int size, const float* class_prob, const int class_num, float* max_prob, int* idx, int *class_idx, const int conf_thresh)
{
int index = blockIdx.x * blockDim.x + threadIdx.x;
if(index < size)
{
// printf("run here %d!\n", index);
float temp_max_prob = 0.0f;
const float *start = class_prob + index * class_num;
int class_index = -1;
for(int i = 0; i < class_num; i++)
{
float curr_prob = start[i];
if(temp_max_prob <= curr_prob)
{
class_index = i;
temp_max_prob = curr_prob;
}
}
max_prob[index] = 0.0f;
if(temp_max_prob >= conf_thresh)
{
// atomicAdd(detecNum, 1);
max_prob[index] = temp_max_prob;
// printf("run here %d!\n", index);
}
idx[index] = index;
class_idx[index] = class_index;
}
}