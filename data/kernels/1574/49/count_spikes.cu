#include "includes.h"
__global__ void count_spikes(const double *Params, const int *id, int *nsp, const float *x, float *V){

int tid, tind, bid, ind, Nspikes, Nfilters, NthreadsMe, Nblocks;

Nspikes               = (int) Params[0];
Nfilters             = (int) Params[2];

tid 		= threadIdx.x;
bid 		= blockIdx.x;
NthreadsMe              = blockDim.x;
Nblocks               = gridDim.x;

tind = tid + NthreadsMe *bid;

while (tind<Nfilters){
for(ind=0; ind<Nspikes;ind++)
if (id[ind]==tind){
nsp[tind] ++;
V[tind] += x[tind];
}
V[tind] = V[tind] / (.001f + (float) nsp[tind]);

tind += NthreadsMe * Nblocks;
}


}