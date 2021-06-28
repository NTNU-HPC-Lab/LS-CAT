#include "includes.h"
__global__ void histo_atomic(unsigned int *out_histo,const float *d_in, int num_bins, int size,float min_val,float range)
{
int tid = threadIdx.x;
int id = tid + blockIdx.x * blockIdx.x;
if(tid >= size)
{
return;
}
int bin = ((d_in[id]-min_val)*num_bins)/range;
bin = bin == num_bins ? num_bins -1 : bin; //max value bin is last bin of the histogram
atomicAdd(&(out_histo[bin]),1);
}