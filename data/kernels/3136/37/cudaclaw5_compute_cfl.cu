#include "includes.h"
__global__ void cudaclaw5_compute_cfl(int idir, int mx, int my, int meqn, int mwaves, int mbc, double dx, double dy, double dt, double *speeds, double* cflgrid)
{
#if 0
# from fortran_source/cudaclaw5_flux2.f */

c     # compute maximum wave speed for checking Courant number:
cfl1d = 0.d0
do 50 mw=1,mwaves
do 50 i=1,mx+1
c          # if s>0 use dtdx1d(i) to compute CFL,
c          # if s<0 use dtdx1d(i-1) to compute CFL:
cfl1d = dmax1(cfl1d, dtdx1d(i)*s(mw,i),
&                          -dtdx1d(i-1)*s(mw,i))
50       continue
#endif
/* Compute largest waves speeds, scaled by dt/dx,  on grid */


}