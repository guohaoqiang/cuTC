__device__
void tc_1dim_64_8X8_NN(float* A, int* dim_A, int* inter_A, int* acc_dims_A, int nd_A, \
                    float* B, int* dim_B, int* inter_B, int* acc_dims_B, int nd_B, \
                    float* C, int* dim_C_A, int* acc_dims_C_A, int nd_C_A, int* dim_C_B, int* acc_dims_C_B, int nd_C_B, int Mask, \
                    int64_t* dim, int nd){
    
    // A: {dim[dim_A[0]], dim[dim_A[1]], dim[dim_A[2]], ...}
    // B: {dim[dim_B[0]], dim[dim_B[1]], dim[dim_B[2]], ...}
    // C: {dim[dim_C[0]], dim[dim_C[1]], dim[dim_C[2]], ...}
    int BY = 4, BX = 4, BK = 8;
    extern __shared__ float sh[]; 
    // shared_mem: BY*BK + BX*BK | BY*BK + BX*BK
    float *sh_A = sh;
	float *sh_B = sh + (BY*BK);  // maybe needs to modified
//------------------------------C---------------------------    
    // C (a,b  X  c,d)
    // horizontal
    int blocks_x = (dim[dim_C_B[0]] + BX - 1)/BX;
    
    // vertical
	int blocks_y = (dim[dim_C_A[0]] + BY - 1)/BY;
 
    // Load C from global memory to register file
    // Here, blockDim.x = BX * BY
    //                   + a                                               + b * a_dims                     + c * a_dims * b_dims                                               + d * a_dims * b_dims * c_dims
	//float *C_start = C + (blockIdx.y % blocks_y * BY + threadIdx.x % BY) + blockIdx.y / blocks_y * a_dims + (blockIdx.x % blocks_x * BX + threadIdx.x / BY) * a_dims * b_dims + (blockIdx.x / blocks_x) * a_dims * b_dims * c_dims ;
	int idx[8]; // len is nd, which is the same with dim array
    // BY <= dim[dim_C_A[0]]
    int coeff_y = blockIdx.y % blocks_y * BY + threadIdx.x % BY; 
    // BX <= dim[dim_C_B[0]]
    int coeff_x = blockIdx.x % blocks_x * BX + threadIdx.x / BY;
    
    float reg_C; 
    int C_offset = 0;
    if ( coeff_y < dim[dim_C_A[0]] && coeff_x < dim[dim_C_B[0]] ){
        idx[dim_C_A[0]] = coeff_y;
        idx[dim_C_B[0]] = coeff_x;

        C_offset += idx[dim_C_A[0]];
        for (int i=1; i<nd_C_A; ++i){
            idx[dim_C_A[i]] = blockIdx.y / (blocks_y * acc_dims_C_A[i-1]);
            C_offset += idx[dim_C_A[i]] * acc_dims_C_A[i];
        }

        C_offset += idx[dim_C_B[0]] * acc_dims_C_B[0] * acc_dims_C_A[nd_C_A];
        for (int i=1; i<nd_C_B; ++i){
            idx[dim_C_B[i]] = blockIdx.x / (blocks_x * acc_dims_C_B[i-1]);
            C_offset += idx[dim_C_B[i]] * acc_dims_C_B[i] * acc_dims_C_A[nd_C_A];
        }
        //C_start = C;
        reg_C = C[C_offset]; 
    }    
//----------------------------------------------------------    
//------------------------------A---------------------------    
    // A (a e b f)
    //load A from global memory to shared memory
    //                   + a                                               + b * a_dims * e_dims                     + (e) 0 * a_dims  + (f) 0 * a_dims * e_dims * b_dims
    //var/folders/6_/_434ff9j2cz1f0psvgwdf9700000gn/T/TemporaryItems/NSIRD_screencaptureui_XG5uhF/Screen\ Shot\ 2022-01-06\ at\ 2.46.30\ PM.png /float *A_start = A + (blockIdx.y % blocks_y * BY + threadIdx.x % BY) + blockIdx.y / blocks_y * a_dims * e_dims + 0 * a_dims      + threadIdx.x / BY * a_dims * e_dims * b_dims;
    //int A_base = A + (blockIdx.y % blocks_y * BY + threadIdx.x % BY) + blockIdx.y / blocks_y * a_dims * e_dims;   // external indices
    //float *A_start = A_base + 0 * a_dims      + (threadIdx.x / BY+0) * a_dims * e_dims * b_dims;    // internal indices of A
    int A_internal_offset = 0;
    int A_base = 0;
    
    // the first dim of A is an external dim. A1 case
    for (int i=0; i<nd_A-1; ++i){
        // the dim belongs to external dims
        A_base += idx[dim_A[i]]*acc_dims_A[i];
    }
    // the dim belongs to internal dims
    int d = threadIdx.x / BY;
    if (d < BK){
        // <(less than) internal dims
        A_internal_offset += d * acc_dims_A[nd_A-1];
    }
    A_internal_offset += A_base;
    if ( threadIdx.x < BY*BK ){
        *(sh_A + threadIdx.x) = A[A_internal_offset];
//----------------------------------------------------------    
//------------------------------B---------------------------    
    // B (f c e d)
    //load B from global memory to shared memory
    //                 + c * f_dims                                               +  d * f_dims * c_dims *e_dims                      + (e) 0 * f_dims * c_dims    + (f) 0
    //float *B_start = B + (blockIdx.x % blocks_x * BX + threadIdx.x / BY) * f_dims + (blockIdx.x / blocks_x) * f_dims * c_dims *e_dims + 0 * f_dims * c_dims        + threadIdx.x % BK;
    //int B_base = B + (blockIdx.x % blocks_x * BX + threadIdx.x / BY) * f_dims + (blockIdx.x / blocks_x) * f_dims * c_dims *e_dims;   // external indices of B
    //float *B_start = B_base + 0 * f_dims * c_dims        + threadIdx.x % BK * 1;     // internal indices of B
    
    int B_internal_offset = 0;
    int B_base = 0;
    // the first dim of B is a contraction dim. B1 case
    // in this case, we have select it as the first iteration contraction dim previously
    if (threadIdx.x / BK < BX){
        B_internal_offset += threadIdx.x % BK;
        for (int i=1; i<nd_B; ++i){
            // all the remaining dims are external dims
            B_base += idx[dim_B[i]]*acc_dims_B[i];
        }
    }
    B_internal_offset += B_base;
    if (threadIdx.x < BX*BK){
        *(sh_B + threadIdx.x) = B[B_internal_offset];
    }
//----------------------------------------------------------    
}
