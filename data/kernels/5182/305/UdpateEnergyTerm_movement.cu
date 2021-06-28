#include "includes.h"
__global__ void UdpateEnergyTerm_movement( float* energy, int energy_dim, int nPatches, float * desc, int desc_dim, int id_desc_move) // whic hindex is the one with movement
{

int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
int idDim   = id % energy_dim;
int idPatch = id / energy_dim;
if (id<energy_dim*nPatches){
if (idDim==1) // movement
energy[id] = -desc[idPatch*desc_dim + id_desc_move];
}
}