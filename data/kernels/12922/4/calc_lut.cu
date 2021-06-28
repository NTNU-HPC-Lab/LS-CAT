#include "includes.h"
__global__ void calc_lut(int *lut, int * hist_in, int img_size, int nbr_bin){


__shared__ int shared_hist[256];
shared_hist[threadIdx.x] = hist_in[threadIdx.x];
__syncthreads();
__shared__  int cdf[256];
__syncthreads();

int i, min, d;
//int cdf = 0;
min = 0;
i = 0;

while(min == 0){
min = shared_hist[i++];
}
d = img_size - min;
for (unsigned int stride = 1; stride <= threadIdx.x; stride *= 2) {
__syncthreads();
shared_hist[threadIdx.x] += shared_hist[threadIdx.x-stride];
}
cdf[threadIdx.x] = shared_hist[threadIdx.x];
//printf("cdf = %d\n",cdf);
__syncthreads();



//for(i = 0; i <= threadIdx.x; i ++){	//tha mporouse na ginei me prefix sum san veltistoipohsh FIXME
//  cdf += shared_hist[i];
//  lut[i] = (cdf - min)*(nbr_bin - 1)/d;
//}
//printf("cdf = %d\n",cdf);


lut[threadIdx.x] = (int)(((float)cdf[threadIdx.x] - min)*255/d + 0.5);
if(lut[threadIdx.x] < 0){
lut[threadIdx.x] = 0;
}
}