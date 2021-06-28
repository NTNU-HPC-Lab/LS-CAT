#define CUDASTDOFFSET threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z))
//new series 
