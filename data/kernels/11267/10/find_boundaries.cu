#include "includes.h"

__global__ void find_boundaries(const int num_keys, const int num_bucket, const int *which_bucket, int *bucket_start){
int index = threadIdx.x + blockIdx.x*blockDim.x +blockIdx.y*blockDim.x*gridDim.x;
// Each thread looks at one entry in the sorted bucket index list
if (index >= num_keys){
return;
}
int previous_bucket = (index > 0 ? which_bucket[index - 1] : 0);
int my_bucket = which_bucket[index];
/*
*/
if (previous_bucket != my_bucket){
for (int i = previous_bucket; i < my_bucket; ++i){
bucket_start[i] = index;
}
}

/*
*/
if (index == num_keys - 1){
for (int i = my_bucket; i < num_bucket; ++i){
bucket_start[i] = num_keys;
}
}
}