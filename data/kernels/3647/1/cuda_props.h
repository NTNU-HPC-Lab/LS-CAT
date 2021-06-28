#ifndef _CUDA_PROPS_H
#define _CUDA_PROPS_H

cudaDeviceProp get_deviceProps();
// For use on device
#define WARP_SIZE 32
// For use on host
int get_warpSize();
int get_maxThreadsPerBlock();
int get_maxThreadsDim(int);
int get_sharedMemPerBlock();

#endif // _CUDA_PROPS_H defined
