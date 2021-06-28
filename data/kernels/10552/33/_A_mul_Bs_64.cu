#include "includes.h"
__global__ void _A_mul_Bs_64(int mx, int ns, double *x, double *sval, int *srow, int *scol, double *k) {
int s0, s1, sp, sc, sr, x0, xr, k0, k1, kp;
double sv, xv;
sc = threadIdx.x + blockIdx.x * blockDim.x;
while (sc < ns) {	// sc: 0-based column for s and k to be processed
k0 = mx*sc;		// k[k0]: first element of k[:,sc]
k1 = k0+mx;		// k[k1-1]: last element of k[:,sc]
for (kp = k0; kp < k1; kp++) k[kp] = 0;
s0 = scol[sc]-1;    // first element of s[:,sc] is at sval[s0] (scol entries are 1-based)
s1 = scol[sc+1]-1;  // last element of s[:,sc] is at sval[s1-1]
for (sp = s0; sp < s1; sp++) {
sr = srow[sp]-1;  // sr: 0-based row for s (srow entries are 1-based)
sv = sval[sp];	// sv: s[sr,sc] (0-based), this value multiplies the sr'th column of x
x0 = mx*sr;	// x[x0]: first element of x[:,sr]
for (xr = 0; xr < mx; xr++) {
xv = x[x0+xr];     // xv: x[xr,sr], now we can set k[xr,sc]
k[k0+xr] += xv*sv;
}
}
sc += blockDim.x * gridDim.x;
}
}