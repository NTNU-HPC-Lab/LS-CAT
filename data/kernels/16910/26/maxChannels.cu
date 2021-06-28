#include "includes.h"
const int  Nthreads = 1024, maxFR = 10000, NrankMax = 3, nt0max=81, NchanMax = 17;

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void maxChannels(const double *Params, const float *dataraw, const float *data, const int *iC, int *st, int *id, int *counter){

int nt0, indx, tid, tid0, i, bid, NT, Nchan, NchanNear,j,iChan, nt0min;
double Cf, d;
float spkTh;
bool flag;

NT 		= (int) Params[0];
Nchan     = (int) Params[1];
NchanNear = (int) Params[2];
nt0       = (int) Params[3];
nt0min    = (int) Params[4];
spkTh    = (float) Params[5];

tid 		= threadIdx.x;
bid 		= blockIdx.x;

tid0 = tid + bid * blockDim.x;
while (tid0<NT-nt0-nt0min){
for (i=0; i<Nchan;i++){
iChan = iC[0 + NchanNear * i];
Cf    = (double) data[tid0 + NT * iChan];
flag = true;

for(j=1; j<NchanNear; j++){
iChan = iC[j+ NchanNear * i];
if (data[tid0 + NT * iChan] > Cf){
flag = false;
break;
}
}

if (flag){
iChan = iC[NchanNear * i];
if (Cf>spkTh){
d = (double) dataraw[tid0+nt0min-1 + NT*iChan]; //
if (d > Cf-1e-6){
// this is a hit, atomicAdd and return spikes
indx = atomicAdd(&counter[0], 1);
if (indx<maxFR){
st[indx] = tid0;
id[indx] = iChan;
}
}
}
}
}
tid0 += blockDim.x * gridDim.x;
}
}