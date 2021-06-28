#include "includes.h"



using namespace std;

#define D 3
#define N 200
#define K 512
#define Nt 20
#define Rt 0.1f
#define c 0.001f
#define ct 0.0001f




__global__ void addcuda(float* Q, float* P, float* Qt, float* Pt, float* Eg, float* Epg) {
for (int j = 0; j < 10; j++) {
int x = blockIdx.x;
int y = threadIdx.x;
int i = x * K * D + y * D;

float Px = P[i + 0];
float Py = P[i + 1];
float Pz = P[i + 2];
float E = Eg[i/3];
float Ep = Epg[i/3];

float Qx = Q[i + 0];
float Qy = Q[i + 1];
float Qz = Q[i + 2];

float nQx = Q[i + 0] + c * P[i + 0];
float nQy = Q[i + 1] + c * P[i + 1];
float nQz = Q[i + 2] + c * P[i + 2];

// Îòðàæåíèå îò ñòåíîê îáëàñòè

if ((nQx > 1) || (nQx < 0)) {
Px = (-1) * Px;
}
if ((nQy > 1) || (nQy < 0)) {
Py = (-1) * Py;
}
if ((nQz > 1) || (nQz < 0)) {
Pz = (-1) * Pz;
}

// Îòðàæåíèå îò òóðáóëåíòíîñòåé

for (int nt = 0; nt < Nt; nt += 1) {
float Range = (sqrt(pow(Qx - Qt[nt + 0], 2) + pow(Qy - Qt[nt + 1], 2) + pow(Qz - Qt[nt + 2], 2)));
float nRange = (sqrt(pow(nQx - Qt[nt + 0], 2) + pow(nQy - Qt[nt + 1], 2) + pow(nQz - Qt[nt + 2], 2)));

if((Range > Rt) && (nRange < Rt)) {
float DirX = (nQx - Qt[nt + 0]) / Range;
float DirY = (nQy - Qt[nt + 1]) / Range;
float DirZ = (nQz - Qt[nt + 2]) / Range;
float PnormKoe = ((Px * DirX) + (Py * DirY) + (Pz * DirZ));
float Pnormt = ((Pt[nt + 0] * DirX) + (Pt[nt + 1] * DirY) + (Pt[nt + 2] * DirZ));
E -= (ct / c) * (PnormKoe * PnormKoe) * (Pnormt * abs(Pnormt));
Px -= 2 * DirX;
Py -= 2 * DirY;
Pz -= 2 * DirZ;
}
}
// ×àñòèöà âûëåòàåò èç îáëàñòè, çàïèñûâàåòñÿ åå ýíåðãèÿ è ñáðàñûâàåòñÿ äî íà÷àëüíîãî çíà÷åíèÿ.
// ×àñòèöà ïðîäîëæàåò äâèãàòüñÿ ïî òðàåêòîðèè
// Ep ñëó÷àéíàÿ âåëè÷èíà ëèíåéíî çàâèñÿùàÿ îò ýíåðãèè
if ((nQz > 1) && (E > Ep)) {
E = 100.0f;
}
// Àäèàáàòè÷åñêîå îõëàæäåíèå
if (nQz > 0.5) {
E -= 0.0001f;
}
//Ïðèðàùåíèå ýíåðãèè ïðè ïåðåñå÷åíèè öåíòðà
if (((nQz > 0.5f) && (Qz < 0.5f)) || ((Qz > 0.5f) && (nQz < 0.5f))) {
E += 1.0f;
}

// Çàïèñü â ïàìÿòü
Q[i + 0] = nQx;
Q[i + 1] = nQy;
Q[i + 2] = nQz;

P[i + 0] = Px;
P[i + 1] = Py;
P[i + 2] = Pz;
Eg[i/3] = E;
}
}