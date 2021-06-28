#include "includes.h"
/*
* SpaceTime Simulator
*   Curso Deep Learning y Cuda - 2020
*   Autor: Oscar Noel Amaya Garcia
*   email: dbanshee@gmail.com
*/



#define RUN_MODE_SIM            0
#define RUN_MODE_BENCH          1

#define SP_FILENAME             "sp.json"
#define SP_FILENAME_BUFF1       "sp_0.json"
#define SP_FILENAME_BUFF2       "sp_1.json"
#define SP_FILENAME_BENCH       "sp_bench.json"
#define MAX_BLACK_HOLES         10
#define SOFTENING               1e-9f
#define DT                      0.05f

#define MAX_ASTEROIDS           30
#define AST_FILENAME            "ast.json"
#define AST_FILENAME_BUFF1      "ast_0.json"
#define AST_FILENAME_BUFF2      "ast_1.json"
#define AST_FILENAME_BENCH      "ast_bench.json"

#define MAX_BENCHMARKS          128
#define BENCH_FILENAME          "benchmark.json"
#define BENCH_TIME_SECS         10
#define BENCH_CPU               0
#define BENCH_GPU               1
#define BENCH_REGEN_BH_STEPS    5
#define BENCH_FILE_ACCESS_STEPS 3

#define CUDA_OPT_NLEVELS        4
#define CUDA_OPT_LEVEL_0        0
#define CUDA_OPT_LEVEL_1        1
#define CUDA_OPT_LEVEL_2        2
#define CUDA_OPT_LEVEL_3        3

#define MAX_TIME_SIMULATION_SEC 360
#define REGEN_BLACK_HOLES_SEC   20

#define NUM_BECHMARKS 10

typedef struct blackHole {
float x, y, g;
} blackHole;

typedef struct spacePoint {
float x, y, g;
} spacePoint;

typedef struct asteroid {
float x, y, vx, vy;
} asteroid;


typedef struct benchmark {
char name[1024];
int number;
int config;
int type;   // CPU = 0, GPU = 1
long time;  // millis
int steps;
} benchmark;


///////////////
// Global Vars
///////////////

// Runtime
int runMode = RUN_MODE_SIM;
int spCurrentBuff = 0;
int astCurrentBuff = 0;
int nBlackHoles = 0;
int nAsteroids = MAX_ASTEROIDS;
blackHole* blackHoles = NULL;
int bhSize;
asteroid* asteroids = NULL;
int astSize;
spacePoint* SPBox = NULL;
int spSize;
float top = 2, left = -2, bottom = -2, right = 2;
float spStep = 0.1;
int nelems;
int rows, cols;
int cudaOptLevel = CUDA_OPT_LEVEL_3;

// BenchMark
int nBenchmark;
int bechmarkRegenBHSteps = MAX_TIME_SIMULATION_SEC;
int bechmarkRegenWriteFileSteps = REGEN_BLACK_HOLES_SEC;
benchmark BENCHS[MAX_BENCHMARKS];
char benchName[1024];
int benchNum;
int benchConfig;
int benchType;


//////////////////
// Error Handling
//////////////////

__global__ void calculateAsteroidsKSHM(asteroid* AST, int nAsteroids, blackHole* BH, int nBlackHoles, float dt, int bhSize) {

extern __shared__ float s[];
blackHole* bhCache = (blackHole*) s;

int idx    = (blockIdx.x * blockDim.x) + threadIdx.x;
int stride = gridDim.x * blockDim.x;

if (idx == 0)
memcpy(bhCache, BH, bhSize);
__syncthreads();

for (int i = idx; i < nAsteroids; i += stride) {
float Fx = 0.0f; float Fy = 0.0f;

for (int j = 0; j < nBlackHoles; j++) {
float dx = AST[i].x - bhCache[j].x;
float dy = AST[i].y - bhCache[j].y;

if (dx == 0) break;
if (dy == 0) break;

float distSqr = dx*dx + dy*dy;
float invDist = 1 / sqrt(distSqr*distSqr);

Fx += dx * invDist; Fy += dy * invDist;
}

float modF = sqrt(Fx*Fx + Fy*Fy);
Fx = Fx / modF;
Fy = Fy / modF;

AST[i].vx += dt*-Fx;
AST[i].vy += dt*-Fy;
AST[i].x  += AST[i].vx*0.01;
AST[i].y  += AST[i].vy*0.01;
}
}