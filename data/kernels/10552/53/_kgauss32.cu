#include "includes.h"
__global__ void _kgauss32(int mx, int ns, float *xval, int *xrow, int *xcol, float *sval, int *srow, int *scol, float g, float *k) {
// assume x(mx,nd) and s(nd,ns) are in 1-based csc format
// assume k(mx,ns) has been allocated and zeroed out
int s0, s1, sp, sc, sr, x0, x1, xp, xc, xr, k0, k1, kp;
float sv, xv, xs;
sc = threadIdx.x + blockIdx.x * blockDim.x;
k0 = mx*sc;		// k[k0]: first element of k[:,sc]
k1 = k0+mx;		// k[k1-1]: last element of k[:,sc]
while (sc < ns) {	// sc: 0-based column for s
s0 = scol[sc]-1;    // first element of s[:,sc] is at sval[s0] (scol entries are 1-based)
s1 = scol[sc+1]-1;  // last element of s[:,sc] is at sval[s1-1]
for (sp = s0; sp < s1; sp++) {
sr = srow[sp]-1;  // sr: 0-based row for s (srow entries are 1-based)
sv = sval[sp];	// sv: s[sr,sc] (0-based)
xc = sr;		// xc: 0-based column for x (=sr)
x0 = xcol[xc]-1;  // first element of x[:,xc] is at xval[x0]
x1 = xcol[xc+1]-1; // last element of x[:,xc] is at xval[x1-1]
for (xp = x0; xp < x1; xp++) {
xr = xrow[xp]-1; // xr: 0-based row for x
xv = xval[xp];	 // xv: x[xr,xc=sr], now we can set k[xr,sc]
xs = xv - sv;
k[k0+xr] += xs*xs; // k += (xi-si)^2
}
}
for (kp = k0; kp < k1; kp++) {
k[kp] = exp(-g*k[kp]); // k = exp(-g*sum((xi-si)^2))
}
sc += blockDim.x * gridDim.x;
}
}