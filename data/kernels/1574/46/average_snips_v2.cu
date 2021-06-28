#include "includes.h"
__global__ void average_snips_v2(const double *Params, const int *iC, const int *call, const int *id, const float *uproj, const float *cmax, float *WU){


// jic, version with no threading over features, to avoid
// collisions when summing WU
// run

int my_chan, this_chan, bid, ind, Nspikes, NrankPC, NchanNear, Nchan;
float xsum = 0.0f;
int chanIndex, pcIndex;

Nspikes               = (int) Params[0];
NrankPC             = (int) Params[1];
Nchan                = (int) Params[7];
NchanNear             = (int) Params[6];


bid 		= blockIdx.x;

for(ind=0; ind<Nspikes;ind++)
if (id[ind]==bid){
my_chan = call[ind];
for (chanIndex = 0; chanIndex < NchanNear; ++chanIndex) {
this_chan = iC[chanIndex + NchanNear * my_chan];
for (pcIndex = 0; pcIndex < NrankPC; ++pcIndex) {
xsum = uproj[pcIndex + NrankPC*chanIndex +  NrankPC*NchanNear * ind];
WU[pcIndex + NrankPC*this_chan + NrankPC*Nchan * bid] +=  xsum;
}
}

}
}