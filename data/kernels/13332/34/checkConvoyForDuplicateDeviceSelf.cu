#include "includes.h"
__global__ void checkConvoyForDuplicateDeviceSelf(Convoy* d_convoy, bool* d_duplicate)
{
//check whether thread is in bounds
if(((threadIdx.x < d_convoy[blockIdx.x].endIndexTracks)  && (d_convoy[blockIdx.x].endIndexTracks > d_convoy[blockIdx.x].startIndexTracks)) || ((d_convoy[blockIdx.x].endIndexTracks < d_convoy[blockIdx.x].startIndexTracks) && (threadIdx.x != d_convoy[blockIdx.x].endIndexTracks)))
{
d_duplicate[blockIdx.x] = true;
bool result = (d_convoy[blockIdx.x].tracks[threadIdx.x].x != 0.5);
if(!result)
{
d_duplicate[blockIdx.x] = d_duplicate[blockIdx.x] && result;
}
}
}