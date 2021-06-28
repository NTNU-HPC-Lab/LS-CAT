#include "includes.h"
__global__ void resampleFeaturesKernel(double* u, double* v, double* d, double* vu, double* vv, double* vd, double* weights, double* randvals, int n_features, double* u_sampled, double* v_sampled, double* d_sampled, double* vu_sampled, double* vv_sampled, double* vd_sampled)
{
// each block corresponds to 1 feature. there may be more features
// than the maximum number of blocks, so we use this for loop

int n_particles = blockDim.x ;

for ( int n = blockIdx.x ; n < n_features; n += gridDim.x ){
double interval = 1.0/n_particles ;
double r = randvals[n] + threadIdx.x*interval ;

int offset = blockDim.x*n ;
double c = weights[offset] ;
int idx = offset ;
while ( r > c ){
c += weights[++idx] ;

if (idx == offset + n_particles){
idx-- ;
break ;
}
}

int idx_new = n*blockDim.x + threadIdx.x ;
u_sampled[idx_new] = u[idx] ;
v_sampled[idx_new] = v[idx] ;
d_sampled[idx_new] = d[idx] ;
vu_sampled[idx_new] = vu[idx] ;
vv_sampled[idx_new] = vv[idx] ;
vd_sampled[idx_new] = vd[idx] ;
}
}