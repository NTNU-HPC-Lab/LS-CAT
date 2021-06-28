#include "includes.h"
__global__ void findIDInConvoyDevice(Convoy* d_convoy, int* d_IDIncluded, int id1, int id2)
{
if(((threadIdx.x < d_convoy[blockIdx.x].endIndexID)  && (d_convoy[blockIdx.x].endIndexID > d_convoy[blockIdx.x].startIndexID)) || ((d_convoy[blockIdx.x].endIndexID < d_convoy[blockIdx.x].startIndexID) && (threadIdx.x != d_convoy[blockIdx.x].endIndexID)))
{
int index = blockIdx.x*2;
d_IDIncluded[index] = INT_MAX;
d_IDIncluded[index+1] = INT_MAX;
__syncthreads();
int result = (d_convoy[blockIdx.x].participatingVehicles[threadIdx.x] == id1);
if(result)
{
atomicMin(&(d_IDIncluded[index]), threadIdx.x);
}
result = (d_convoy[blockIdx.x].participatingVehicles[threadIdx.x] == id2);
if(result)
{
atomicMin(&(d_IDIncluded[index+1]), threadIdx.x);
}
result = (d_convoy[blockIdx.x].participatingVehicles[threadIdx.x] == -1);
if(result)
{
atomicMin(&(d_IDIncluded[index+1]), INT_MIN);
atomicMin(&(d_IDIncluded[index]), INT_MIN);
}

}
}