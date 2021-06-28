#include "includes.h"
__global__ void rotate(float*a,float b, float * c, int sx,int sy,int sz, int dx, int dy, int dz, int ux, int uy, int uz)
{
int id=(blockIdx.x*blockDim.x+threadIdx.x); // id of this processor

int Processes=blockDim.x * gridDim.x;
int chains=ux*uy*uz; // total number of independent chains
int N=sx*sy*sz;  // total size of array, has to be chains*length_of_chain
int length=N/chains;  // chain length
int steps=N/Processes;  // this is how many steps each processor has to do

int step,nl,nx,ny,nz,x,y,z,i,idd;
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