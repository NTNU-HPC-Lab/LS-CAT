#include "includes.h"
__global__ void  bestFilter(const double *Params, const float *data, const float *mu, const float *lam, const float *nu, float *xbest, float *err, int *ftype){

int tid, tid0, i, bid, NT, Nfilt, ibest = 0;
float Th,  Cf, Ci, xb, Cbest = 0.0f, epu, cdiff;

tid 		= threadIdx.x;
bid 		= blockIdx.x;
NT 		= (int) Params[0];
Nfilt 	= (int) Params[1];
Th 		= (float) Params[2];
epu       = (float) Params[8];

tid0 = tid + bid * Nthreads;
if (tid0<NT-1 & tid0>0){
for (i=0; i<Nfilt;i++){
Ci = data[tid0 + NT * i] + mu[i] * lam[i];
Cf = Ci * Ci / (lam[i] + 1.0f) - lam[i]*mu[i]*mu[i];

// add the shift component
cdiff = data[tid0+1 + NT * i] - data[tid0-1 + NT * i];
Cf = Cf + cdiff * cdiff / (epu + nu[i]);
if (Cf > Cbest){
Cbest 	= Cf;
xb      = Ci  - mu[i] * lam[i]; /// (lam[i] + 1);
ibest 	= i;
}
}
if (Cbest > Th*Th){
err[tid0] 	= Cbest;
xbest[tid0] 	= xb;
ftype[tid0] 	= ibest;
}
}
}