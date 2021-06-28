#include "includes.h"


//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////



//////////////////////////////////////////////////////////////////////////////////////////







//////////////////////////////////////////////////////////////////////////////////////////
__global__ void average_snips(const double *Params, const int *iC, const int *call, const int *id, const float *uproj, const float *cmax, float *WU){

int my_chan, this_chan, tidx, tidy, bid, ind, Nspikes, NrankPC, NchanNear, Nchan;
float xsum = 0.0f;

Nspikes               = (int) Params[0];
NrankPC             = (int) Params[1];
Nchan                = (int) Params[7];
NchanNear             = (int) Params[6];

tidx 		= threadIdx.x;
tidy 		= threadIdx.y;
bid 		= blockIdx.x;

for(ind=0; ind<Nspikes;ind++)
if (id[ind]==bid){
my_chan = call[ind];
this_chan = iC[tidy + NchanNear * my_chan];
xsum = uproj[tidx + NrankPC*tidy +  NrankPC*NchanNear * ind];
WU[tidx + NrankPC*this_chan + NrankPC*Nchan * bid] +=  xsum;
}
}