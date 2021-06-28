#ifndef LABELMETHOD_H
#define LABELMETHOD_H

#include <cstdlib>
// Created by Torben Trindkaer Nielsen, 1 Oct 2014
// http://www.codeproject.com/Articles/825200/An-Implementation-Of-The-Connected-Component-Label

#define CALL_LabelComponent(x,y,returnLabel) { STACK[SP] = x; STACK[SP+1] = y; STACK[SP+2] = returnLabel; SP += 3; goto START; }
#define RETURN { SP -= 3;                \
                 switch (STACK[SP+2])    \
                 {                       \
                 case 1 : goto RETURN1;  \
                 case 2 : goto RETURN2;  \
                 case 3 : goto RETURN3;  \
                 case 4 : goto RETURN4;  \
                 default: return;        \
                 }                       \
               }
#define XLAB (STACK[SP-3])
#define YLAB (STACK[SP-2])


void LabelComponent(unsigned short* STACK, unsigned short width,
        unsigned short height, int* input, int* output, int labelNo,
        unsigned short x, unsigned short y);

// Returns the labelled image as well as the number of individual components found
int LabelImage(unsigned short width, unsigned short height, int* input,
        int* output);

#endif
