#include "includes.h"


using namespace std;


#define ULL unsigned long long

const long MAXDIM = 10;
const double RMIN = 2.0;
const double RMAX = 7.0;

#define MAX_THREADS 1024
#define MAX_BLOCKS 65535


//Global kernel code that runs on the device

__global__ void count_in(ULL *dev_count, long dev_ntotal,long dev_ndim, long dev_halfb, double dev_rsquare, long dev_base){

//Calculate the position of this kernel in the data
ULL blockID = (blockIdx.y * gridDim.x) + blockIdx.x;
ULL pos = (blockID * blockDim.x) + threadIdx.x;

//If this threads position in the data is further than we need to calculate
//Then we return
if(pos >= dev_ntotal) return;

double rtestsq = 0;
long idx = 0;
long index[MAXDIM+1];
for (long i = 0; i < dev_ndim; ++i) index[i] = 0;

//Convert the decimal number into another base system
while (pos != 0) {
long rem = pos % dev_base;
pos = pos / dev_base;
index[idx] = rem;
++idx;
}


for(long k = 0; k < dev_ndim; ++k){
double xk = index[k] - dev_halfb;
rtestsq += xk * xk;
}

//If the value is inside the sphere
//Atomically add 1 to the count
if(rtestsq < dev_rsquare){
atomicAdd(dev_count, 1);
}
}