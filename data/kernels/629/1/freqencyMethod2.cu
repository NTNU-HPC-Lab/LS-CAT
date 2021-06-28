#include "includes.h"
/*
Detected 1 CUDA Capable device(s)

Device 0: "GeForce GT 320M"
CUDA Driver Version / Runtime Version          5.0 / 5.0
CUDA Capability Major/Minor version number:    1.2
Total amount of global memory:                 1024 MBytes (1073741824 bytes)
( 3) Multiprocessors x (  8) CUDA Cores/MP:    24 CUDA Cores
GPU Clock rate:                                1100 MHz (1.10 GHz)
Memory Clock rate:                             790 Mhz
Memory Bus Width:                              128-bit
Max Texture Dimension Size (x,y,z)             1D=(8192), 2D=(65536,32768), 3D=(2048,2048,2048)
Max Layered Texture Size (dim) x layers        1D=(8192) x 512, 2D=(8192,8192) x 512
Total amount of constant memory:               65536 bytes
Total amount of shared memory per block:       16384 bytes
Total number of registers available per block: 16384
Warp size:                                     32
Maximum number of threads per multiprocessor:  1024
Maximum number of threads per block:           512
Maximum sizes of each dimension of a block:    512 x 512 x 64
Maximum sizes of each dimension of a grid:     65535 x 65535 x 1
Maximum memory pitch:                          2147483647 bytes
Texture alignment:                             256 bytes
Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
Run time limit on kernels:                     Yes
Integrated GPU sharing Host Memory:            No
Support host page-locked memory mapping:       Yes
Alignment requirement for Surfaces:            Yes
Device has ECC support:                        Disabled
CUDA Device Driver Mode (TCC or WDDM):         WDDM (Windows Display Driver Model)
Device supports Unified Addressing (UVA):      No
Device PCI Bus ID / PCI location ID:           2 / 0
Compute Mode:
< Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 5.0, CUDA Runtime Version = 5.0, NumDevs = 1, Device0 = GeForce GT 320M
*/




__global__ void freqencyMethod2(char *d_dat,int len, int *d_freq)
{//·½·¨¶þ£¬ÏÈ½«Êý¾ÝÔ­×Ó¼Óµ½share memoryÖÐ£¬È»ºóÔÙÀÛ¼Óµ½ÏÔ´æÉÏ¡£

__shared__ int sfreq[26];//

if(threadIdx.x < 26)
sfreq[threadIdx.x] = 0;////ÏÈÇå¿Õ¡£
__syncthreads();
int gridsize = blockDim.x * gridDim.x;
int pos = 0;
for(int i=threadIdx.x + blockIdx.x*blockDim.x; i< len; i += gridsize)
{
pos = d_dat[i]-'a';
atomicAdd(&sfreq[pos],1);
}
__syncthreads();

if(threadIdx.x<26)///Èç¹ûÏÔ¿¨Ö§³ÖÔ­×Ó¼Ó£¬¿ÉÒÔÊ¹ÓÃÔ­×Ó¼Ó£¬Ö±½Ó¼Óµ½ÏÔ´æÉÏ¡£ÄÇÑù¾ÍÃ»ÓÐµÚ¶þ²½¡£ 1.1¼°ÒÔÉÏÖ§³ÖÈ«¾ÖÏÔ´æµÄ32Î»Ô­×Ó²Ù×÷¡£
atomicAdd(&d_freq[threadIdx.x],sfreq[threadIdx.x]);

}