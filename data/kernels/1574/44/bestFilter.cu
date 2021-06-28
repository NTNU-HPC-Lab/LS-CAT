#include "includes.h"
__global__ void bestFilter(const double *Params,  const bool *match, const int *iC, const int *call, const float *cmax, int *id, float *cx){

int Nchan, tid,tind,bid, ind, Nspikes, Nfilters, Nthreads, Nblocks, my_chan;
float max_running = 0.0f;

Nspikes               = (int) Params[0];
Nfilters              = (int) Params[2];
Nthreads              = blockDim.x;
Nblocks               = gridDim.x;
Nchan                = (int) Params[7];

tid 		= threadIdx.x;
bid 		= blockIdx.x;

tind = tid + bid * Nthreads;

while (tind<Nspikes){
max_running = 0.0f;
id[tind] = 0;
my_chan = call[tind];

for(ind=0; ind<Nfilters; ind++)
if (match[my_chan + ind * Nchan])
if (cmax[tind + ind*Nspikes] > max_running){
id[tind] = ind;
max_running = cmax[tind + ind*Nspikes];
}


cx[tind] = max_running;

tind += Nblocks*Nthreads;
}
}