#include "includes.h"


//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////



//////////////////////////////////////////////////////////////////////////////////////////







//////////////////////////////////////////////////////////////////////////////////////////
__global__ void computeCost(const double *Params, const float *uproj, const float *mu, const float *W, const bool *match, const int *iC, const int *call, float *cmax){

int NrankPC,j, NchanNear, tid, bid, Nspikes, Nthreads, k, my_chan, this_chan, Nchan;
float xsum = 0.0f, Ci, lam;

Nspikes               = (int) Params[0];
NrankPC             = (int) Params[1];
Nthreads              = blockDim.x;
lam                   = (float) Params[5];
NchanNear             = (int) Params[6];
Nchan                 = (int) Params[7];

tid 		= threadIdx.x;
bid 		= blockIdx.x;

while(tid<Nspikes){
my_chan = call[tid];
if (match[my_chan + bid * Nchan]){
xsum = 0.0f;
for (k=0;k<NchanNear;k++)
for(j=0;j<NrankPC;j++){
this_chan = iC[k + my_chan * NchanNear];
xsum += uproj[j + NrankPC * k + NrankPC*NchanNear * tid] *
W[j + NrankPC * this_chan +  NrankPC*Nchan * bid];
}
Ci = max(0.0f, xsum) + lam/mu[bid];

cmax[tid + bid*Nspikes] = Ci * Ci / (1.0f + lam/(mu[bid] * mu[bid])) - lam;
}
tid+= Nthreads;
}
}