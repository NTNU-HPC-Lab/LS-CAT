#include "includes.h"
__global__ void sum_dWU(const double *Params, const float *bigArray, float *WU) {

int tid,bid, ind, Nfilters, Nthreads, Nfeatures, Nblocks, NfeatW, nWU, nElem;
float sum = 0.0f;

Nfeatures             = (int) Params[1];  //NrankPC, number of pcs
NfeatW                = (int) Params[4];  //Nchan*nPC
Nfilters              = (int) Params[2];
Nthreads              = blockDim.x;
Nblocks               = gridDim.x;

tid 		= threadIdx.x;
bid 		= blockIdx.x;


//WU is NfeatW x Nfilters.

nWU = NfeatW * Nfilters;
nElem = Nfeatures*NfeatW; //number of elements in each subArray of bigArray

//Calculate which element we're addressing
int tind = tid + bid * Nthreads;

int currFilt, currFW, currIndex;
while (tind < nWU){


//which filter and element of WU?
currFilt = floor((double)(tind/NfeatW));
currFW = tind - currFilt*NfeatW;

//Sum up the Nfeature elements of bigArray that correspond to this
//filter and NfeatW

sum = 0.0f;

for(ind=0; ind<Nfeatures; ind++) {
//bigArray is Nfilter arrays of Nfeature x NfeatW;
currIndex = currFilt*nElem + ind*NfeatW + currFW;
sum += bigArray[ currIndex ];
}

WU[tind] += sum;
tind += Nblocks*Nthreads;

}

}