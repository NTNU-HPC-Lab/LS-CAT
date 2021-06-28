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


__global__ void rotate(float*a,float b, float * c, size_t sx,size_t sy,size_t sz, size_t dx, size_t dy, size_t dz, size_t ux, size_t uy, size_t uz)
{
// id of this processor
size_t id=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x);

size_t Processes=blockDim.x * gridDim.x;
size_t chains=ux*uy*uz; // total number of independent chains
size_t N=sx*sy*sz;  // total size of array, has to be chains*length_of_chain
size_t length=N/chains;  // chain length
size_t steps=N/Processes;  // this is how many steps each processor has to do

size_t step,nl,nx,ny,nz,x,y,z,i,idd;
float swp, nswp;

//if (id != 0)   return;
//for (id=0;id<Processes;id++)
{
step=steps*id;   // my starting step as the id times the number of steps
nl=step%length;  // current position in chain length
nx=(step/length)%ux;  // current position in unit cell x
ny=(step/(length*ux))%uy;  // current position in unit cell y
nz=(step/(length*ux*uy))%uz;  // current position in unit cell z
i=0;

//if (step/steps != 4 && step/steps != 5) return;

while(nz<uz)
{
while(ny<uy)
{
while (nx<ux)
{
x=(nx+nl*dx)%sx;  // advance by the offset steps along the chain
y=(ny+nl*dy)%sy;
z=(nz+nl*dz)%sz;
idd=x+sx*y+sx*sy*z;
if (i < steps) {
swp=a[idd];
// a[idd]=a[idd]+0.1;
__syncthreads();
}
while (nl<length-1)
{
if (i > steps-1)
goto nextProcessor; // return;
if (step >= N)  // this thread has reached the end of the total data to process
goto nextProcessor; // return;
step++;
x = (x+dx)%sx; // new position
y = (y+dy)%sy;
z = (z+dz)%sz;
idd=x+sx*y+sx*sy*z;
if (i < steps-1) {
nswp=a[idd];
__syncthreads();
//a[idd]=a[idd]+0.1;
}

c[idd]=swp+0.1; // c[idd]+ny+0.1; // c[idd]+i; // swp+0.1; // c[idd]+(step/steps);
i++; // counts number of writes
if (i > steps-1)
goto nextProcessor; // return;
nl++;
if (i < steps) {
swp=nswp;
}
}
nx++; nl=0;
//if (nx < ux) {
x = (x+dx)%sx; // new position
y = (y+dy)%sy;
z = (z+dz)%sz;
idd=x+sx*y+sx*sy*z;
c[idd]=swp+0.1; // no need to save this value as this is the end of the line
//}
i++;
if (i > steps-1)
goto nextProcessor; // return;
// if (nx <ux) x=(x+1)%sx;
}
ny++;
// if (ny <uy) y=(y+1)%sy;
nx=0;x=0;
}
nz++;
// if (nz <uz) z=(z+1)%sz;
ny=0;y=0;
}
nextProcessor:
nz=0;
}
return;
}