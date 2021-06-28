#include "includes.h"
__global__ void average_snips(const double *Params, const int *ioff, const int *id, const float *uproj, const float *cmax, const int *iList, float *cf, float *WU){

int tid, bid, ind, Nspikes, Nfeatures, NfeatW, Nnearest, t;
float xsum = 0.0f, pm;

Nspikes               = (int) Params[0];
Nfeatures             = (int) Params[1];
pm                    = (float) Params[3];
NfeatW                = (int) Params[4];
Nnearest              = (int) Params[6];

tid 		= threadIdx.x;
bid 		= blockIdx.x;

for(ind=0; ind<Nspikes;ind++)
if (id[ind]==bid){

xsum = uproj[tid + Nfeatures * ind];
WU[tid + ioff[ind] + NfeatW * bid] = pm * WU[tid + ioff[ind] + NfeatW * bid]
+ (1-pm) * xsum;

// go through the top 10 nearest filters and match them
for (t=0;t<Nnearest;t++)
cf[ind + t*Nspikes] = cmax[ind + Nspikes * iList[t + Nnearest*bid]];

}
}