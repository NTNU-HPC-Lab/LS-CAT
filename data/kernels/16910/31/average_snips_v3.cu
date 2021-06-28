#include "includes.h"


//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////



//////////////////////////////////////////////////////////////////////////////////////////







//////////////////////////////////////////////////////////////////////////////////////////
__global__ void average_snips_v3(const double *Params, const int *ioff, const int *id, const float *uproj, const float *cmax, float *bigArray){


// jic, version to work with Nfeatures threads
// have made a big array of Nfeature*NfeatW*Nfilters so projections
// onto each Nfeature can be summed without collisions
// after running this, need to sum up each set of Nfeature subArrays
// to calculate the final NfeatW*Nfilters array

int tid, bid, ind, Nspikes, Nfeatures, NfeatW;
float xsum = 0.0f;

Nspikes               = (int) Params[0];
Nfeatures             = (int) Params[1];
NfeatW                = (int) Params[4];

tid       = threadIdx.x;      //feature index
bid 		= blockIdx.x;       //filter index





for(ind=0; ind<Nspikes;ind++) {

if (id[ind]==bid){
//uproj is Nfeatures x Nspikes
xsum = uproj[tid + Nfeatures * ind];
//add this to the Nfeature-th array of NfeatW at the offset for this spike
bigArray[ioff[ind] + tid + tid*NfeatW + Nfeatures*NfeatW * bid] +=  xsum;
}  //end of if block for  match
}     //end of loop over spikes

}