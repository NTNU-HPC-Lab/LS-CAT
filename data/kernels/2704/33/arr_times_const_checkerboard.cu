#include "includes.h"
/************************* CudaMat ******************************************
*   Copyright (C) 2008-2009 by Rainer Heintzmann                          *
*   heintzmann@gmail.com                                                  *
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; Version 2 of the License.               *
*                                                                         *
*   This program is distributed in the hope that it will be useful,       *
*   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
*   GNU General Public License for more details.                          *
*                                                                         *
*   You should have received a copy of the GNU General Public License     *
*   along with this program; if not, write to the                         *
*   Free Software Foundation, Inc.,                                       *
*   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
***************************************************************************
* Compile with:
* Windows:
system('"c:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin\vcvars32.bat"')
system('nvcc -c cudaArith.cu -ccbin "c:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin')

Window 64 bit:
system('nvcc -c cudaArith.cu -ccbin "c:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin" -I"c:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\include" ')

Linux:
* File sudo vi /usr/local/cuda/bin/nvcc.profile
* needs the flag -fPIC  in the include line
system('nvcc -c cudaArith.cu -v -I/usr/local/cuda/include/')
*/

// To suppress the unused variable argument for ARM targets
#pragma diag_suppress 177


#ifndef NAN   // should be part of math.h
#define NAN (0.0/0.0)
#endif

#define ACCU_ARRTYPE double  // Type of the tempory arrays for reduce operations
#define IMUL(a, b) __mul24(a, b)

//#define BLOCKSIZE 512
//#define BLOCKSIZE 512
// below is blocksize for temporary array for reduce operations. Has to be a power of 2 in size
#ifndef CUIMAGE_REDUCE_THREADS  // this can be defined at compile time via the flag NVCCFLAG='-D CUIMAGE_REDUCE_THREADS=512'
#define CUIMAGE_REDUCE_THREADS 512
#endif
// (prop.maxThreadsPerBlock)
// #define CUIMAGE_REDUCE_THREADS 512
// #define CUIMAGE_REDUCE_THREADS 128
//#define CUIMAGE_REDUCE_BLOCKS  64

#define NBLOCKS(N,blockSize) (N/blockSize+(N%blockSize==0?0:1))

#define NBLOCKSL(N,blockSize) 1
// min((N/blockSize+(N%blockSize==0?0:1)),prop.maxGridSize[0])


__global__ void arr_times_const_checkerboard(float*a,float b, float * c, size_t N, size_t sx,size_t sy,size_t sz)
{
size_t ids=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x); // which source array element do I have to deal with?
if(ids>=N) return;  // not in range ... quit

size_t px=(ids/2)%sx;   // my x pos
size_t py=(ids/2)/sx;   // my y pos
float minus1=(1-2*((px+py)%2));
c[ids]=a[ids]*b*minus1;
}