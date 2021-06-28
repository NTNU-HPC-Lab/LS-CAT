#include "includes.h"
__global__ void computeCost(const double *Params, const float *Ws, const float *mus, const float *W, const float *mu, const bool *iMatch, const int *iC, const int *Wh, float *cmax){

int j, tid, bid, Nspikes, my_chan, this_chan, Nchan, NrankPC, NchanNear, Nthreads, k;
float xsum = 0.0f, Ci;

Nspikes               = (int) Params[0];  //more accurately, number of comparisons, Nfilt*Nbatch
Nchan                 = (int) Params[7];
NrankPC                 = (int) Params[1];
NchanNear                 = (int) Params[6];
Nthreads              = blockDim.x;


tid 		= threadIdx.x;
bid 		= blockIdx.x;

while(tid<Nspikes){
my_chan = Wh[tid];
if (iMatch[my_chan + bid*Nchan]){
xsum = 0.0f;
for (k=0;k<NchanNear;k++){
this_chan = iC[k + NchanNear * my_chan];
for (j=0;j<NrankPC;j++)
xsum += Ws[j + NrankPC*k + NrankPC*NchanNear * tid] *
W[j + NrankPC*this_chan + NrankPC*Nchan * bid];

}

Ci = mu[bid]*mu[bid] + mus[tid]*mus[tid] -2*mus[tid]*mu[bid]*xsum;
cmax[tid + bid*Nspikes] = Ci;
}
tid+= Nthreads;
}
}