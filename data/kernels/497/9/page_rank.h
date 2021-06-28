#pragma once
const int BUFFER_SIZE = (16 * 1024);
const char TOEKNS[3] = ": ";
const int INT_SIZE = sizeof(int);
#define DECAY 0.85f
#define THRESHOLD 30
#define BLOCK_SIZE 1024

struct CSC_st
{
	// size: num vertices+1
	int* destination_offsets;
	// size: num edges
	int* source_indices;
	int nvertices; int nedges;
	int* out_degrees;
};
typedef struct CSC_st *CSC_t;
