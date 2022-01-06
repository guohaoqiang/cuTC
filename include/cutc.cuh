#ifndef CUTC_H
#define CUTC_H 
#include "tensor.h"

#include <stdlib.h>
#include <stdio.h>

void tensorContraction_host(float A[], int dim_A[], int nd_A, \
                   float B[], int dim_B[], int nd_B, \
                    float C[], int32_t dim_C[], int nd_C, \
                    int64_t dim[], int nd);
void run_cuTC(Tensor &AA, Tensor &BB, Tensor &CC);
#endif /* CUTC_H */
