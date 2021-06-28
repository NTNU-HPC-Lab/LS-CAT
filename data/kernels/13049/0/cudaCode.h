/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   cudaCode.h
 * Author: INovitskas
 *
 * Created on 14 мая 2016 г., 14:24
 */

#ifndef CUDACODE_H
#define CUDACODE_H

void cudaSets(int threadID);
void cudaRun(int* Data, unsigned int DataSize,int i);
int* setCudaData(int* CData);

#endif /* CUDACODE_H */

