#include "includes.h"
__global__ void calc_lut(int *lut, int * hist_in, int img_size, int nbr_bin){


__shared__ int shared_hist[256];
shared_hist[threadIdx.x] = hist_in[threadIdx.x];
__syncthreads();

int i, cdf, min, d;
cdf = 0;
min = 0;
i = 0;

while(min == 0){
min = shared_hist[i++];
}
d = img_size - min;
for(i = 0; i <= threadIdx.x; i ++){	//tha mporouse na ginei me prefix sum san veltistoipohsh FIXME
cdf += shared_hist[i];
//lut[i] = (cdf - min)*(nbr_bin - 1)/d;
}

lut[threadIdx.x] = (int)(((float)cdf - min)*255/d + 0.5);
if(lut[threadIdx.x] < 0){
lut[threadIdx.x] = 0;
}
}