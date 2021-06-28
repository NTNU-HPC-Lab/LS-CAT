#include "includes.h"
__global__ void count_spikes(const double *Params, const int *id, int *nsp){

int tid, tind, bid, ind, Nspikes, Nfilters, Nthreads, Nblocks;

Nspikes               = (int) Params[0];
Nfilters             = (int) Params[2];

tid 		= threadIdx.x;
bid 		= blockIdx.x;
Nthreads              = blockDim.x;
Nblocks               = gridDim.x;

tind = tid + Nthreads *bid;

while (tind<Nfilters){
for(ind=0; ind<Nspikes;ind++)
if (id[ind]==tind)
nsp[tind] += 1;
tind += Nthreads * Nblocks;
}
}