#pragma once
#ifndef STEEL_H
#define STEEL_H
#include <string>
#include <map>
#include <stdlib.h> 
using namespace std;
class Steel
{
public:
    float pho;
    float ce;
    float lamda;
    float temperature_l, temperature_s;
    map<string, float> * components;
    Steel(map<string, float> *, float lamda, float pho, float ce);
    void print();
};
#endif