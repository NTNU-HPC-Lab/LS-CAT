#include "includes.h"
__global__ void DecompressionKernel(int dimensionalityd, unsigned char *compressed_data_buffer_in, int *chunk_boundaries_buffer_in, unsigned long long *uncompressed_data_buffer_out) {
register int offset, code, bcount, off, beg, end, lane, warp, iindex, lastidx, start, term;
register unsigned long long diff, prev;
__shared__ int ibufs[32 * (3 * WARPSIZE / 2)];


// index within this warp
lane = threadIdx.x & 31;
// index within shared prefix sum array
iindex = threadIdx.x / WARPSIZE * (3 * WARPSIZE / 2) + lane;
ibufs[iindex] = 0;
iindex += WARPSIZE / 2;
lastidx = (threadIdx.x / WARPSIZE + 1) * (3 * WARPSIZE / 2) - 1;
// warp id
warp = (threadIdx.x + blockIdx.x * blockDim.x) / WARPSIZE;
// prediction index within previous subchunk
offset = WARPSIZE - (dimensionalityd - lane % dimensionalityd) - lane;

// determine start and end of chunk to decompress
start = 0;
if (warp > 0)
start = chunk_boundaries_buffer_in[warp - 1];
term = chunk_boundaries_buffer_in[warp];
off = ((start + 1) / 2 * 17);

prev = 0;
for (int i = start + lane; i < term; i += WARPSIZE) {
// read in half-bytes of size and leading-zero count information

if ((lane & 1) == 0) {
code = compressed_data_buffer_in[off + (lane >> 1)];

//4352
// printf(" %i ", start);
return;
ibufs[iindex] = code; //THIS line is crashing
return;
ibufs[iindex + 1] = code >> 4;

}
return;
off += (WARPSIZE / 2);
__threadfence_block();
code = ibufs[iindex];

bcount = code & 7;
if (bcount >= 2)
bcount++;

// calculate start positions of compressed data
ibufs[iindex] = bcount;
__threadfence_block();
ibufs[iindex] += ibufs[iindex - 1];
__threadfence_block();
ibufs[iindex] += ibufs[iindex - 2];
__threadfence_block();
ibufs[iindex] += ibufs[iindex - 4];
__threadfence_block();
ibufs[iindex] += ibufs[iindex - 8];
__threadfence_block();
ibufs[iindex] += ibufs[iindex - 16];
__threadfence_block();

// read in compressed data (the non-zero bytes)
beg = off + ibufs[iindex - 1];
off += ibufs[lastidx];
end = beg + bcount - 1;
diff = 0;
for (; beg <= end; end--) {
diff <<= 8;
diff |= compressed_data_buffer_in[end];
}

// negate delta if sign bit indicates it was negated during compression
if ((code & 8) != 0) {
diff = -diff;
}

// write out the uncompressed word
uncompressed_data_buffer_out[i] = prev + diff;
__threadfence_block();

// save prediction for next subchunk
prev = uncompressed_data_buffer_out[i + offset];
}
}