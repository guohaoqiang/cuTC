#ifndef MYCUTENSOR_H
#define MYCUTENSOR_H 
#include "tensor.h"

#include <stdlib.h>
#include <stdio.h>

#include <unordered_map>
#include <vector>

//#include "/usr/local/cuda-10.2/include/cuda_runtime.h"
//#include "/home/ghq/2021/cuTC/libcutensor/include/cutensor.h"

void run_cuTensor(Tensor& A, Tensor& B, Tensor& C);

#endif /* MYCUTENSOR_H */
