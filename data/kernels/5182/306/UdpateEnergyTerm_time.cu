#include "includes.h"
__global__ void UdpateEnergyTerm_time( float* energy, int energy_dim, int nPatches, float * idFocuser_focused , float par_time_increase_energy_on_focus, float par_time_decrease_energy_in_time)
{

int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
int idDim   = id % energy_dim;
int idPatch = id / energy_dim;
if (id<energy_dim*nPatches){
if (idDim==0){ // time
if (idPatch==(int)(*idFocuser_focused)) // it is id that focuser just focused
energy[id] += par_time_increase_energy_on_focus;
else
energy[id] /= par_time_decrease_energy_in_time ;
}
}
}