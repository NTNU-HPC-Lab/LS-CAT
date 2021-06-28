#define threadindex ( ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x )
//new series 
