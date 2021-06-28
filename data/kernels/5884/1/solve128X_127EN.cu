#include "includes.h"
/// System includes

// CUDA runtime

#define threadsPerBlock  (512)
#define MaxCuckooNum (4*4096)
#define MaxGpuNum (1024)
#define trim (32)
#define SolveThreadsPerBlock (128)
#define SolveEN (128)
#define CuckooNum (2*4096)

#define rotl(x, b) (((x) << (b)) | ((x) >> (64 - (b))))
#define EBIT 15
#define CLEN 12
#define EN (1 << EBIT)
#define M (EN << 1)
#define MASK ((1 << EBIT) - 1)
#define CN CLEN << 2

struct GPU_DEVICE
{
uint32_t cproof[CuckooNum][CLEN];
uint8_t msg[CuckooNum][32];
uint8_t alive[CuckooNum][EN >> 3];
uint8_t calive[CuckooNum][EN >> 3];
uint64_t nonces[CuckooNum];

uint8_t  *gmsg = NULL;
uint8_t  *gRHash = NULL;
uint32_t *gRege = NULL;
uint32_t *gproof = NULL;
uint32_t *gnode = NULL;
};

GPU_DEVICE *gpu_divices[MaxGpuNum] = {NULL};
uint32_t gpu_divices_cnt = 0;

// set siphash keys from 32 byte char array
#define setkeys() \
k0 = (((uint64_t *)mesg)[0]); \
k1 = (((uint64_t *)mesg)[1]); \
k2 = (((uint64_t *)mesg)[2]); \
k3 = (((uint64_t *)mesg)[3]);

#define sip_round() \
v0 += v1; v2 += v3; v1 = rotl(v1, 13); \
v3 = rotl(v3, 16); v1 ^= v0; v3 ^= v2; \
v0 = rotl(v0, 32); v2 += v1; v0 += v3; \
v1 = rotl(v1, 17); v3 = rotl(v3, 21); \
v1 ^= v2; v3 ^= v0; v2 = rotl(v2, 32);

__global__ void solve128X_127EN(uint32_t *gRege, uint8_t *gRHash, uint32_t *gproof)
{
unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
uint32_t i, tmp;
uint8_t u, v;

uint32_t block_tid = id % SolveThreadsPerBlock;
uint32_t *Rege = gRege + id * SolveEN;
uint8_t *RHash = gRHash + id * (SolveEN << 1);
uint32_t *proof = gproof + id * CLEN;

__shared__ uint32_t path[SolveThreadsPerBlock][CLEN];
__shared__ uint8_t graph[SolveThreadsPerBlock][SolveEN << 1];

uint8_t pre;
uint8_t cur;
uint8_t next;

memset(&graph[block_tid], 0xff, (SolveEN << 1));
proof[0] = 0xffffffff;

for (i = 0; i<SolveEN; i++)
{
if (Rege[i] == 0xffffffff)
{
break;
}
u = RHash[i << 1];
v = RHash[(i << 1) + 1];
__syncthreads();
pre = 0xff;
cur = u;
while (cur != 0xff)
{
next = graph[block_tid][cur];
graph[block_tid][cur] = pre;
pre = cur;
cur = next;
}
int m = 0;
cur = v;
while (graph[block_tid][cur] != 0xff && m < CLEN)
{
cur = graph[block_tid][cur];
++m;
}
if (cur != u)
{
graph[block_tid][u] = v;
}
else if (m == CLEN - 1)
{
int j;
cur = v;
for (j = 0; j <= m; ++j)
{
path[block_tid][j] = cur;
cur = graph[block_tid][cur];
}

memset(&graph[block_tid], 0xff, (SolveEN << 1));

for (j = 1; j <= m; ++j)
{
graph[block_tid][path[block_tid][j]] = path[block_tid][j - 1];
}

int k = 0;
int b = CLEN - 1;
for (j = 0; k < b; ++j)
{
u = RHash[j << 1];
v = RHash[(j << 1) + 1];
if (graph[block_tid][u] == v)
{
path[block_tid][k] = Rege[j];
graph[block_tid][u] = 0xff;
++k;
}
else if(graph[block_tid][v] == u)
{
path[block_tid][k] = Rege[j];
graph[block_tid][v] = 0xff;
++k;
}
}
path[block_tid][k] = Rege[i];

for (j = 0; j < CLEN - 1; j++) // sort
{
for (k = 0; k < CLEN - j - 1; k++)
{
if (path[block_tid][k]>path[block_tid][k + 1])
{
tmp = path[block_tid][k];
path[block_tid][k] = path[block_tid][k + 1];
path[block_tid][k + 1] = tmp;
}
}
}
for (j = 0; j < CLEN; j++)proof[j] = path[block_tid][j];
break;
}
}
__syncthreads();
}