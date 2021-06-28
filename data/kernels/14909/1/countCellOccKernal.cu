#include "includes.h"
//----------------------------------------------------------------------------------------------------------------------
/// @file CudaSPHKernals.cu
/// @author Declan Russell
/// @date 08/03/2015
/// @version 1.0
//----------------------------------------------------------------------------------------------------------------------

#define pi 3.14159265359f

//----------------------------------------------------------------------------------------------------------------------
/// @brief Kernal designed to produce a has key based on the location of a particle
/// @brief Hash function taken from Teschner, M., Heidelberger, B., Mueller, M., Pomeranets, D. and Gross, M.
/// @brief (2003). Optimized spatial hashing for collision detection of deformable objects
/// @param d_hashArray - pointer to a buffer to output our hash keys
/// @param d_posArray - pointer to the buffer that holds our particle positions
/// @param numParticles - the number of particles in our buffer
/// @param resolution - the resolution of our hash table
/// @param _gridScaler - Scales our points to between 0-1.
//----------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------
/// @brief This kernal is designed to count the cell occpancy of a hash table
/// @param d_hashArray - pointer to hash table buffer
/// @param d_cellOccArray - output array of cell occupancy count
/// @param _hashTableSize - the size of our hash table
/// @param _numPoints - the number of particles in our hashed array
//----------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------
/// @brief This is our desity weighting kernal used in our navier stokes equations
/// @param _dst - the distance away of the neighbouring
/// @param _smoothingLength - the smoothing length of our simulation. Can be thought of a hash cell size.
/// @param _densKernCosnt - constant part of our kernal. Easier to calculate once on CPU and have loaded into device kernal.
/// @return return the weighting that our neighbouring particle has on our current particle
//----------------------------------------------------------------------------------------------------------------------
__global__ void countCellOccKernal(unsigned int *d_hashArray, unsigned int *d_cellOccArray, int _hashTableSize, unsigned int _numPoints){
//Create our idx
int idx = threadIdx.x + blockIdx.x * blockDim.x;

// Make sure our idx is valid and add the occupancy count to the relevant cell
if ((idx < _numPoints) && (d_hashArray[idx] < _hashTableSize)) {
atomicAdd(&(d_cellOccArray[d_hashArray[idx]]), 1);
}
}