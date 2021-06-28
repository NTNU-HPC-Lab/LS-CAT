#include "includes.h"
__global__ void bestFilter(const double *Params, const bool *iW, const float *cmax, int *id){

int tid,tind,bid, ind, Nspikes, Nfilters, Nthreads, Nblocks;
float max_running = 0.0f, Th;

Nspikes               = (int) Params[0];
Nfilters              = (int) Params[2];
Nthreads              = blockDim.x;
Nblocks               = gridDim.x;
Th                    = (float) Params[7];

tid 		= threadIdx.x;
bid 		= blockIdx.x;

tind = tid + bid * Nthreads;

while (tind<Nspikes){
max_running = 0.0f;
id[tind] = 0;

for(ind=0; ind<Nfilters; ind++)
if (iW[tind + ind*Nspikes])
if (cmax[tind + ind*Nspikes] > max_running){
id[tind] = ind;
max_running = cmax[tind + ind*Nspikes];
}

if (max_running < Th*Th)
id[tind] = -1;

tind += Nblocks*Nthreads;
}
}