#include "includes.h"
const int  Nthreads = 1024, maxFR = 100000, NrankMax = 3, nmaxiter = 500, NchanMax = 32;
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

// THIS UPDATE DOES NOT UPDATE ELOSS?
//////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////






//////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void average_snips(const double *Params, const int *st, const int *id,  const float *x, const float *y,  const int *counter, const float *dataraw, const float *W, const float *U, double *WU, int *nsp, const float *mu, const float *z){

int nt0, tidx, tidy, bid, NT, Nchan,k, Nrank, Nfilt;
int currInd;
float Th;
double  X, xsum;

NT        = (int) Params[0];
Nfilt    	=   (int) Params[1];
nt0       = (int) Params[4];
Nrank     = (int) Params[6];
Nchan     = (int) Params[9];

tidx 		= threadIdx.x;
bid 		= blockIdx.x;

//Th = 10.f;
Th 		= (float) Params[15];

// we need wPCA projections in here, and then to decide based on total

// idx is the time sort order of the spikes; the original order is a function
// of when threads complete in mexGetSpikes. Compilation of the sums for WU, sig, and dnextbest
// in a fixed order makes the calculation deterministic.

for(currInd=0; currInd<counter[0];currInd++) {
// only do this if the spike is "GOOD"
if (x[currInd]>Th){
if (id[currInd]==bid){
if (tidx==0 &&  threadIdx.y==0)
nsp[bid]++;

tidy 		= threadIdx.y;
while (tidy<Nchan){
X = 0.0f;
for (k=0;k<Nrank;k++)
X += W[tidx + bid* nt0 + nt0*Nfilt*k] *
U[tidy + bid * Nchan + Nchan*Nfilt*k];

xsum = dataraw[st[currInd]+tidx + NT * tidy] + y[currInd] * X;

//WU[tidx+tidy*nt0 + nt0*Nchan * bid] *= p[bid];
WU[tidx+tidy*nt0 + nt0*Nchan * bid] += (double) xsum;

tidy+=blockDim.y;

}        //end of while loop over channels
}               //end of if block for id == bid
}
}                  //end of for loop over spike indicies
}                      //end of function