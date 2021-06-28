#include "includes.h"
#define MAX_STEPS 32


void readFile();
int cpuPathTest(int limitSteps, unsigned long long *tid);
void printMaze();
void printPath(unsigned long long tid, int steps);
void printPathMaze(unsigned long long tid, int steps);
void setTime0();
void getExeTime();

struct Maze
{
char maze[99][99];
int rows, cols, s_x, s_y, e_x, e_y;
};

struct Maze maze;
FILE *MAZE;
struct timespec t_start, t_end;
double elapsedTime;

const int threadsPerBlock = 1024;
const int blocksPerGrid = 1024;



__global__ void testPath(int *limitSteps, struct Maze *maze, int *workDone , unsigned long long *path)
{
unsigned long long tid = threadIdx.x + blockIdx.x * blockDim.x;
unsigned long long bias = blockDim.x * gridDim.x;
unsigned long long maxRoute = 0xffffffffffffffff - (bias - 1);//max length of path
maxRoute >>= (MAX_STEPS - *limitSteps) * 2;//(32 - 1) * 2 = 62 =>0~011 only 3 steps : right up left

while(tid <= maxRoute)
{
if(*workDone) break;

int x = maze->s_x, y = maze->s_y;
unsigned long long temp = tid;

int i = *limitSteps;
int steps = 0;
do
{
//GetMoveDirection
steps++;
int direction = temp & 3;//mask
temp >>= 2;
//Move
switch(direction)
{
case 0 :
x += 1;
break;
case 1 :
y -= 1;
break;
case 2 :
x -= 1;
break;
case 3 :
y += 1;
break;
}
//if at Target, print path ,else keep going, if no way then break
if(maze->maze[y][x] == '$')
{
*workDone = 1;
*path = tid;
break;
}
else if(maze->maze[y][x] != '.')
{
break;
}
}
while(i--);

tid += bias;
}
}