#include "includes.h"
__global__ void rgamma_kernel(curandState *state, int state_len, float *vals, int n, float a, float scale)
{
float sqrt32 = 5.656854;
float exp_m1 = 0.36787944117144232159;
float q1 = 0.04166669;
float q2 = 0.02083148;
float q3 = 0.00801191;
float q4 = 0.00144121;
float q5 = -7.388e-5;
float q6 = 2.4511e-4;
float q7 = 2.424e-4;
float a1 = 0.3333333;
float a2 = -0.250003;
float a3 = 0.2000062;
float a4 = -0.1662921;
float a5 = 0.1423657;
float a6 = -0.1367177;
float a7 = 0.1233795;

// Maybe want each thread to generate multiple...
// -- Would cut down on bloat from curandState *state,
// -- Would increase computation per thread + scalability
//
// state_len    :: length of rng states (one thread per rng_state)
//                 if (idx >= state_len){ do nothing... }
//
// n_per_thread :: number of rng's to generate per thread
//                 heurestically, each thread computes
//                 n_per_thread = n / state_len
//                 variates. However, if n_per_thread is
//                 not an integer, then some do one less,
//                 some one more.
//
// n(idx)       :: number of rng's that thread idx will generate
//
// n_lo         :: minimum number of rng's that a (used) thread
//                 will generate
//
// n_hi         :: maximum number of rng's that a (used) thread
//                 will generate
//
// n_extra      :: (= n % state_len) the number of threads that
//                 will generate n_hi rng's
//
// n_regular    :: (= state_len - n_extra) the number of threads that
//                 will generate n_lo rng's
//
// Note:
//
// n = (n_lo * n_regular) + (n_hi * n_extra)
//
// e.g.,
//
// n = 108, state_len = 40
// => n_lo = 2, n_hi = 3
// => n_extra = 28, n_regular = 12
// => threads  1-28 generate 3 rng's
// => threads 29-40 generate 2 rngs'
//
// thread idx writes to:
//
// if (idx < n_extra){
//   // 'extra' thread:
//   n_hi*idx : ((n_hi*idx) + (n_hi-1))
// }
// if (idx >= n_extra){
//   // 'regular' thread:
//   (n_hi*n_extra) + (n_lo*(idx-n_extra)):((n_lo*(idx-n_extra)) + (n_lo-1))
// }
//
// e.g., (cont...)
//
// thread  0 writes to: 0,1,2
// thread  1 writes to: 3,4,5
// ...
// thread 27 writes to: 81,82,83
// thread 28 writes to: 84,85
// thread 29 writes to: 86,87
// ...
// thread 40 writes to: 118,119
//

// Usual block/thread indexing...
int myblock = blockIdx.x + blockIdx.y * gridDim.x;
int blocksize = blockDim.x * blockDim.y * blockDim.z;
int subthread = threadIdx.z*(blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
int idx = myblock * blocksize + subthread;

if (idx >= state_len) {
return;
}

// Number of regular and extra threads:
int n_extra = (n % state_len);
//int n_regular = n - n_extra;
int n_lo = (int) (n/state_len);
int n_hi = (n_lo+1);
int out_idx_start, n_per_thread;
if (idx < n_extra){
n_per_thread = n_hi;
out_idx_start = n_hi*idx;
} else {
n_per_thread = n_lo;
out_idx_start = n_hi*n_extra + n_lo*(idx-n_extra);
}
int gen_num;

// State variables:
float aa = 0.0;
float aaa = 0.0;
float s, s2, d;
float q0, b, si, c;
float e, p, q, r, t, u, v, w, x, ret_val;

// Generate RNG's for this thread...
for (gen_num=0; gen_num<n_per_thread; gen_num++){

// Note: Error checks on arguments removed!
if (a < 1.0) { // GS algorithm for parameters a < 1
if (a == 0){
vals[out_idx_start+gen_num] = 0.0;
continue;
}
e = 1.0 + exp_m1 * a;
while (1){
p = e * curand_normal(&state[idx]);
if (p >= 1.0) {
x = -logf((e - p)/a);
if (logf(curand_uniform(&state[idx])) <= (1.0 - a) * logf(x)){
break;
}
} else {
x = expf(logf(p) / a);
if (logf(curand_uniform(&state[idx])) <= x){
break;
}
}
}
vals[out_idx_start+gen_num] = scale * x;
continue;
}

// --- a >= 1 : GD algorithm ---

// Step 1: Recalculations of s2, s, d if a has changed
if (a != aa) {
aa = a;
s2 = a - 0.5;
s = sqrtf(s2);
d = sqrt32 - s * 12.0;
}

// Step 2: t = standard normal deviate,
//           x = (s,1/2) -normal deviate.

//printf("Step 2...\n");

// immediate acceptance (i)
t = curand_normal(&state[idx]);
x = s + 0.5 * t;
ret_val = x * x;
if (t >= 0.0){
vals[out_idx_start+gen_num] = scale * ret_val;
continue;
}

//printf("Step 3...\n");

// Step 3: u = 0,1 - uniform sample. squeeze acceptance (s)
u = curand_uniform(&state[idx]);
if ((d*u) <= (t*t*t)){
vals[out_idx_start+gen_num] = scale * ret_val;
continue;
}

//printf("Step 4...\n");

// Step 4: recalculations of q0, b, si, c if necessary
if (a != aaa){
aaa = a;
r = 1.0 / a;
q0 = ((((((q7 * r + q6) * r + q5) * r + q4) * r + q3) * r
+ q2) * r + q1) * r;

// Approximation depending on size of parameter a
// The constants in the expressions for b, si and c
// were established by numerical experiments

if (a <= 3.686) {
b = 0.463 + s + 0.178 * s2;
si = 1.235;
c = 0.195 / s - 0.079 + 0.16 * s;
} else if (a <= 13.022) {
b = 1.654 + 0.0076 * s2;
si = 1.68 / s + 0.275;
c = 0.062 / s + 0.024;
} else {
b = 1.77;
si = 0.75;
c = 0.1515 / s;
}

}

//printf("Step 5...\n");

// Step 5: no quotient test if x not positive
if (x > 0.0) {

// Step 6: calculation of v and quotient q
v = t / (s + s);
if (fabs(v) <= 0.25){
q = q0 + 0.5 * t * t * ((((((a7 * v + a6) * v + a5) * v + a4) * v
+ a3) * v + a2) * v + a1) * v;
} else {
q = q0 - s * t + 0.25 * t * t + (s2 + s2) * log(1.0 + v);
}

// Step 7: quotient acceptance (q)
if (logf(1.0 - u) <= q){
vals[out_idx_start+gen_num] = scale * ret_val;
continue;
}

}

//printf("Step 8...\n");

while (1){

// Step 8: e = standard exponential deviate
//	u =  0,1 -uniform deviate
//	t = (b,si)-double exponential (laplace) sample
e = -logf(curand_uniform(&state[idx])); // Expo(1)
u = curand_uniform(&state[idx]);
u = u + u - 1.0;
if (u < 0.0){
t = b - si * e;
} else {
t = b + si * e;
}
// Step	 9:  rejection if t < tau(1) = -0.71874483771719
if (t >= -0.71874483771719) {
// Step 10:	 calculation of v and quotient q
v = t / (s + s);
if (fabs(v) <= 0.25){
q = q0 + 0.5 * t * t *
((((((a7 * v + a6) * v + a5) * v + a4) * v + a3) * v
+ a2) * v + a1) * v;
} else {
q = q0 - s * t + 0.25 * t * t + (s2 + s2) * log(1.0 + v);
}
// Step 11:	 hat acceptance (h)
// (if q not positive go to step 8)
if (q > 0.0) {
w = expm1f(q); // expm1 is double prec...
//  ^^^^^ original code had approximation with rel.err < 2e-7
// if t is rejected sample again at step 8
if (c * fabs(u) <= w * expf(e - 0.5 * t * t)){
break;
}
}
}

} // repeat .. until  `t' is accepted

//printf("Returning at end...\n");

x = s + 0.5 * t;
vals[out_idx_start+gen_num] = scale * x * x;
continue;

} // end gen_num loop

return;
}