#include "includes.h"
__global__ void computeCost(const double *Params, const float *uproj, const float *mu, const float *W, const int *ioff, const bool *iW, float *cmax){

int tid, bid, Nspikes, Nfeatures, NfeatW, Nthreads, k;
float xsum = 0.0f, Ci, lam;

Nspikes               = (int) Params[0];
Nfeatures             = (int) Params[1];
NfeatW                = (int) Params[4];
Nthreads              = blockDim.x;
lam                   = (float) Params[5];

tid 		= threadIdx.x;
bid 		= blockIdx.x;

while(tid<Nspikes){
if (iW[tid + bid*Nspikes]){
xsum = 0.0f;
for (k=0;k<Nfeatures;k++)
xsum += uproj[k + Nfeatures * tid] * W[k + ioff[tid] +  NfeatW * bid];

Ci = max(0.0f, xsum) + lam/mu[bid];

cmax[tid + bid*Nspikes] = Ci * Ci / (1.0f + lam/(mu[bid] * mu[bid])) - lam;
}
tid+= Nthreads;
}

}