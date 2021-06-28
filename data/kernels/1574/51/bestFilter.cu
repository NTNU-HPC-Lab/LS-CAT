#include "includes.h"
__global__ void bestFilter(const double *Params, const bool *iMatch, const int *Wh, const float *cmax, const float *mus, int *id, float *x){

int tid,tind,bid, my_chan, ind, Nspikes, Nfilters, Nthreads, Nchan, Nblocks;
float max_running = 0.0f;

Nspikes               = (int) Params[0];
Nfilters              = (int) Params[2];
Nchan                 = (int) Params[7];
Nthreads              = blockDim.x;
Nblocks               = gridDim.x;

tid 		= threadIdx.x;
bid 		= blockIdx.x;

tind = tid + bid * Nthreads;

while (tind<Nspikes){
max_running = mus[tind] * mus[tind];
id[tind] = 0;
my_chan = Wh[tind];
for(ind=0; ind<Nfilters; ind++)
if (iMatch[my_chan + ind * Nchan])
if (cmax[tind + ind*Nspikes] < max_running){
id[tind] = ind;
max_running = cmax[tind + ind*Nspikes];
}
x[tind] = max_running;
tind += Nblocks*Nthreads;
}

}