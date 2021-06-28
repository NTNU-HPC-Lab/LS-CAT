#include "includes.h"

/* Error checking */
#define CUDA_ERROR_CHECK
#define CURAND_ERROR_CHECK
#define CUDA_CALL( err) __cudaCall( err, __FILE__, __LINE__ )
#define CURAND_CALL( err) __curandCall( err, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR()    __cudaCheckError( __FILE__, __LINE__ )

__global__ void initialSpikeIndCopyKernel( unsigned short* pLastSpikeInd, const unsigned int noReal)
{
unsigned int globalIndex = threadIdx.x+blockDim.x*blockIdx.x;
unsigned int spikeNo = globalIndex / noReal;
if (globalIndex<noReal*noSpikes)
{
pLastSpikeInd[globalIndex] = pLastSpikeInd[spikeNo*noReal];
}
}