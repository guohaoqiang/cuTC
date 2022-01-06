#include <cutensor.h>
#include <cuda_runtime.h>
#include "../../include/cutc.cuh"
#include "../../include/util.cuh"
// A[], (0,4,1,5), (1,3), (1, a_dims, a_dims*e_dims, a_dims*e_dims*b_dims), nd_A = 4
// B[], (5,2,4,3), (0,2), (1, f_dims, f_dims*c_dims, f_dims*c_dims*e_dims), nd_B = 4
// (a_dims,b_dims,c_dims,d_dims,e_dims,f_dims), nd = 6
// C[], (0,1) (2,3), (001111), (1, a_dims, a_dims*b_dims, a_dims*b_dims*c_dims), nd_C = 4
// acc_dims_C_A: (1,a_dims,a_dims*b_dims)   len is nd_C_A + 1
// acc_dims_C_B: (1,c_dims,c_dims*d_dims)   len is nd_C_B + 1
__global__ 
void tc_2dims_64_8X8(float A[], int dim_A[], int inter_A[], int acc_dims_A[], int nd_A, \
                    float B[], int dim_B[], int inter_B[], int acc_dims_B[], int nd_B, \
                    float C[], int dim_C_A[], int acc_dims_C_A[], int nd_C_A, int dim_C_B[], int acc_dims_C_B[], int nd_C_B, int Mask, \
                    int64_t dim[], int nd){
    // A: {dim[dim_A[0]], dim[dim_A[1]], dim[dim_A[2]], ...}
    // B: {dim[dim_B[0]], dim[dim_B[1]], dim[dim_B[2]], ...}
    // C: {dim[dim_C[0]], dim[dim_C[1]], dim[dim_C[2]], ...}
    int BY = 8, BX = 8, BK = 8;
    extern __shared__ float sh[]; 
    // shared_mem: BY*BK + BX*BK | BY*BK + BX*BK
    float *sh_A = sh;
	float *sh_B = sh + (BY*BK);  // maybe needs to modified
    
    // C (a,b  X  c,d)
    // horizontal
    //int blocks_x = (c_dims + BX - 1)/BX;
    int blocks_x = (dim[dim_C_B[0]] + BX - 1)/BX;
    
    // vertical
    //int blocks_y = (a_dims + BY - 1)/BY;
	int blocks_y = (dim[dim_C_A[0]] + BY - 1)/BY;
    
    // Load C from global memory to register file
    // Here, blockDim.x = BX * BY
    //                   + a                                               + b * a_dims                     + c * a_dims * b_dims                                               + d * a_dims * b_dims * c_dims
	//float *C_start = C + (blockIdx.y % blocks_y * BY + threadIdx.x % BY) + blockIdx.y / blocks_y * a_dims + (blockIdx.x % blocks_x * BX + threadIdx.x / BY) * a_dims * b_dims + (blockIdx.x / blocks_x) * a_dims * b_dims * c_dims ;
	int idx[8]; // len is nd, which is the same with dim array
    idx[dim_C_A[0]] = (blockIdx.y % blocks_y * BY + threadIdx.x % BY);
    float *C_start = C + idx[dim_C_A[0]];
    for (int i=1; i<nd_C_A; ++i){
        idx[dim_C_A[i]] = blockIdx.y / (blocks_y * acc_dims_C_A[i-1]);
        C_start += idx[dim_C_A[i]] * acc_dims_C_A[i];
    }
    idx[dim_C_B[0]] = (blockIdx.x % blocks_x * BX + threadIdx.x / BY);
    C_start += idx[dim_C_B[0]] * acc_dims_C_B[0] * acc_dims_C_A[nd_C_A];
    for (int i=1; i<nd_C_B; ++i){
        idx[dim_C_B[i]] = blockIdx.x / (blocks_x * acc_dims_C_B[i-1]);
        C_start += idx[dim_C_B[i]] * acc_dims_C_B[i] * acc_dims_C_A[nd_C_A];
    }
    float reg_C = *C_start; 
    
    // A (a e b f)
    //load A from global memory to shared memory
    //                   + a                                               + b * a_dims * e_dims                     + (e) 0 * a_dims  + (f) 0 * a_dims * e_dims * b_dims
    //float *A_start = A + (blockIdx.y % blocks_y * BY + threadIdx.x % BY) + blockIdx.y / blocks_y * a_dims * e_dims + 0 * a_dims      + threadIdx.x / BY * a_dims * e_dims * b_dims;
    //int A_base = A + (blockIdx.y % blocks_y * BY + threadIdx.x % BY) + blockIdx.y / blocks_y * a_dims * e_dims;   // external indices
    //float *A_start = A_base + 0 * a_dims      + (threadIdx.x / BY+0) * a_dims * e_dims * b_dims;    // internal indices of A
    float *A_start = A;
    int first_internal_dim = -1;
    int A_base = 0;
    if (dim_A[0]==dim_C_A[0]){
        // the first dim of A is an external dim. A1 case
        A_start += idx[dim_A[0]];
        for (int i=1; i<nd_A; ++i){
            // the dim belongs to external dims
            if ( (1<<dim_A[i]) & Mask ) A_base += idx[dim_A[i]]*acc_dims_A[i];
            // the dim belongs to internal dims
            else {
            // the first dimension of B is a contraction dim (f dim of B1 in our case), then we iterate the contraction dim of A first. B1 case
            // A1 * B1
                if (dim_B[0]!=dim_C_B[0]){
                    // f dim of A1
                    if (dim_A[i]==dim_C_B[0]){
                        A_start += threadIdx.x / (BY) * acc_dims_A[i];
                        first_internal_dim = dim_A[i];
                    }
                    // e dim of A1
                    else{
                        A_start += 0 * acc_dims_A[i];
                    }
                }
            // the first dimension of B is not a contraction dim, then we iterate the first contraction dim of A first (e dim of A1 in our case). B2 case
            // A1 * B2
                else{
                    // e dim of A
                    if (first_internal_dim == -1){
                        first_internal_dim = dim_A[i];
                        A_start += threadIdx.x / (BY) * acc_dims_A[i];
                    }
                    // f dim of A
                    else{
                        A_start += 0 * acc_dims_A[i];
                    }
                }
            }  
        }
    }else{
        // the first dim of A is a contraction dim. A2 case
        A_start += threadIdx.x % BK;
        // the first contradiction dim of A2. e dim of A2
        first_internal_dim = dim_A[0]; 
        for (int i=1; i<nd_A; ++i){
            // the dim belongs to external dims
            if ( (1<<dim_A[i]) & Mask ) A_base += idx[dim_A[i]] * acc_dims_A[i];
            // the dim belongs to internal dims
            else {
                // the second contraction dim. f dim of A2
                A_start += 0*acc_dims_A[i];
            }
        }
    }
    A_start += A_base;
    *(sh_A + threadIdx.x) = *(A_start);
    // B (f c e d)
    //load B from global memory to shared memory
    //                 + c * f_dims                                               +  d * f_dims * c_dims *e_dims                      + (e) 0 * f_dims * c_dims    + (f) 0
    //float *B_start = B + (blockIdx.x % blocks_x * BX + threadIdx.x / BY) * f_dims + (blockIdx.x / blocks_x) * f_dims * c_dims *e_dims + 0 * f_dims * c_dims        + threadIdx.x % BK;
    //int B_base = B + (blockIdx.x % blocks_x * BX + threadIdx.x / BY) * f_dims + (blockIdx.x / blocks_x) * f_dims * c_dims *e_dims;   // external indices of B
    //float *B_start = B_base + 0 * f_dims * c_dims        + threadIdx.x % BK * 1;     // internal indices of B
    
    float *B_start = B;
    int B_base = 0;
    if (dim_B[0]!=dim_C_B[0]){
        // the first dim of B is a contraction dim. B1 case
        // in this case, we have select it as the first iteration contraction dim previously
        B_start += threadIdx.x % BK;
        for (int i=1; i<nd_B; ++i){
            // the dim belongs to external dims
            if ( (1<<dim_B[i]) & Mask ) B_base += idx[dim_B[i]]*acc_dims_B[i];
            else{
                B_start += 0;
            } 
        }
    }else{
        // the first dim of B is an external dim. B2 case
        B_start += idx[dim_B[0]]*acc_dims_B[0];
        for (int i=1; i<nd_B; ++i){
            // the dim belongs to external dims
            if ( (1<<dim_B[i]) & Mask ) B_base += idx[dim_B[i]]*acc_dims_B[i];
            else{
                if (dim_B[i]==first_internal_dim){
                    B_start += threadIdx.x / BK;
                }else{
                    B_start += 0*acc_dims_B[i];
                }
            } 
        }

    }
    B_start += B_base;
    *(sh_B + threadIdx.x) = *(B_start);
    
    // compute the number of contraction dims
    int prod = dim[dim_A[inter_A[0]]] * dim[dim_A[inter_A[1]]];
    
    // shared_mem: (BY*BK + BX*BK) * 2
    //        0    ~ (BY-1)*BK: A
    // (BY-1)*BK+1 ~ (BY-1)*BK+(BX-1)*BK: B
    int double_buffer = 0; 
    float reg_A, reg_B;
    for (int i=0; i<prod; i += BK){
        __syncthreads();
        int A_offset = double_buffer + (threadIdx.x%BY);
		int B_offset = double_buffer + (threadIdx.x/BY)*BK;
            
        for (int k=0; k<BK; ++k){
            // read A tile from shared memory to registers
            reg_A = sh_A[A_offset];
            // read B tile from shared memory to registers
            reg_B = sh_B[B_offset];
            // perform FMA
            reg_C = fma(reg_A, reg_B, reg_C);

            A_offset += BY;
			B_offset += 1;
        }
        // BX*BK+BY*BK = 8*8+8*8 = 128
        double_buffer ^= (BX*BK+BY*BK);
        if (i+BK < prod){
            if (dim_C_A[0]==dim_A[0]){
                /*
                if (first_internal_dim==dim_A[inter_A[0]]){
                    // internal indices of A. A1 case
                    //f_A = (threadIdx.x/BY + i + BK) % first_internal_dim;
                    //e_A = (threadIdx.x/BY + i + BK) / first_internal_dim;
                    //               + e * a_dims + f * a_dims * e_dims * b_dims
                    A_start = A_base + (threadIdx.x/BY + i + BK) % dim[first_internal_dim] * acc_dims_A[inter_A[0]] + 
                                (threadIdx.x/BY + i + BK) / dim[first_internal_dim] * acc_dims_A[inter_A[1]]; 
                }else{
                    A_start = A_base + (threadIdx.x/BY + i + BK) % dim[first_internal_dim] * acc_dims_A[inter_A[1]] + 
                                (threadIdx.x/BY + i + BK) / dim[first_internal_dim] * acc_dims_A[inter_A[0]];
                }*/
                // the above if..else.. can be combined as one
                A_start = A + A_base + (threadIdx.x/BY + i + BK) % dim[dim_A[inter_A[1]]] * acc_dims_A[inter_A[1]] + 
                                (threadIdx.x/BY + i + BK) / dim[dim_A[inter_A[0]]] * acc_dims_A[inter_A[0]];
            }else{
                // A2 case
                //f_A = (threadIdx.x%BK + i + BK) % first_internal_dim;
                //e_A = (threadIdx.x%BK + i + BK) / first_internal_dim; 
                A_start = A + A_base + (threadIdx.x/BK + i + BK) % dim[first_internal_dim] * acc_dims_A[inter_A[0]] + 
                            (threadIdx.x/BK + i + BK) / dim[first_internal_dim] * acc_dims_A[inter_A[1]]; 
            }
            *(sh_A + double_buffer + threadIdx.x) = *(A_start);

            if (dim_B[0]==first_internal_dim){
                // internal indices of B. B1 case
                //f_B = (threadIdx.x%BK + i + BK) % first_internal_dim;
                //e_B = (threadIdx.x%BK + i + BK) / first_internal_dim;
                //               + e * f_dims * c_dims + f
                B_start = B + B_base + (threadIdx.x/BK + i + BK) % dim[first_internal_dim] * acc_dims_B[inter_B[0]] + 
                (threadIdx.x/BK + i + BK) / dim[first_internal_dim] * acc_dims_B[inter_B[1]];  
            }else{
                /*
                if (first_internal_dim==dim_B[inter_B[0]]){
                    //f_A = (threadIdx.x/BX + i + BK) % first_internal_dim;
                    //e_A = (threadIdx.x/BX + i + BK) / first_internal_dim;
                    //               + e * f_dims * c_dims + f
                    B_start = B_base + (threadIdx.x/BX + i + BK) % dim[first_internal_dim] * acc_dims_B[inter_B[0]] + 
                                (threadIdx.x/BX + i + BK) / dim[first_internal_dim] * acc_dims_B[inter_B[1]];
                }else{
                    B_start = B_base + (threadIdx.x/BX + i + BK) % dim[first_internal_dim] * acc_dims_B[inter_B[1]] + 
                                (threadIdx.x/BX + i + BK) / dim[first_internal_dim] * acc_dims_B[inter_B[0]];
                }*/
                // the above if..else.. can be combined as one
                B_start = B + B_base + (threadIdx.x/BX + i + BK) % dim[dim_B[inter_B[0]]] * acc_dims_B[inter_B[0]] + 
                                (threadIdx.x/BX + i + BK) / dim[dim_B[inter_B[1]]] * acc_dims_B[inter_B[1]];
            }
            *(sh_B + double_buffer + threadIdx.x) = *(B_start);
        }
    }
    // write C tile from register to global memory
    *C_start = reg_C;
}


// A[], (0,4,1,5), (1,3), (1, a_dims, a_dims*e_dims, a_dims*e_dims*b_dims), nd_A = 4
// B[], (5,2,4,3), (0,2), (1, f_dims, f_dims*c_dims, f_dims*c_dims*e_dims), nd_B = 4
// (a_dims,b_dims,c_dims,d_dims,e_dims,f_dims), nd = 6
// C[], (0,1) (2,3), (001111), (1, a_dims, a_dims*b_dims, a_dims*b_dims*c_dims), nd_C = 4
// acc_dims_C_A: (1,a_dims,a_dims*b_dims)   len is nd_C_A + 1
// acc_dims_C_B: (1,c_dims,c_dims*d_dims)   len is nd_C_B + 1
__global__ 
void tc_1dim_64_8X8(float A[], int dim_A[], int inter_A[], int acc_dims_A[], int nd_A, \
                    float B[], int dim_B[], int inter_B[], int acc_dims_B[], int nd_B, \
                    float C[], int dim_C_A[], int acc_dims_C_A[], int nd_C_A, int dim_C_B[], int acc_dims_C_B[], int nd_C_B, int Mask, \
                    int64_t dim[], int nd){
    // A: {dim[dim_A[0]], dim[dim_A[1]], dim[dim_A[2]], ...}
    // B: {dim[dim_B[0]], dim[dim_B[1]], dim[dim_B[2]], ...}
    // C: {dim[dim_C[0]], dim[dim_C[1]], dim[dim_C[2]], ...}
    int BY = 16, BX = 24, BK = 12;
    extern __shared__ float sh[]; 
    // shared_mem: BY*BK + BX*BK | BY*BK + BX*BK
    float *sh_A = sh;
	float *sh_B = sh + (BY*BK);  // maybe needs to modified
    
    // C (a,b  X  c,d)
    // horizontal
    //int blocks_x = (c_dims + BX - 1)/BX;
    int blocks_x = (dim[dim_C_B[0]] + BX - 1)/BX;
    
    // vertical
    //int blocks_y = (a_dims + BY - 1)/BY;
	int blocks_y = (dim[dim_C_A[0]] + BY - 1)/BY;
    
    // Load C from global memory to register file
    // Here, blockDim.x = BX * BY
    //                   + a                                               + b * a_dims                     + c * a_dims * b_dims                                               + d * a_dims * b_dims * c_dims
	//float *C_start = C + (blockIdx.y % blocks_y * BY + threadIdx.x % BY) + blockIdx.y / blocks_y * a_dims + (blockIdx.x % blocks_x * BX + threadIdx.x / BY) * a_dims * b_dims + (blockIdx.x / blocks_x) * a_dims * b_dims * c_dims ;
	int idx[8]; // len is nd, which is the same with dim array
    // BY <= dim[dim_C_A[0]]
    int coeff_y = blockIdx.y % blocks_y * BY + threadIdx.x % BY; 
    // BX <= dim[dim_C_B[0]]
    int coeff_x = (blockIdx.x % blocks_x * BX + threadIdx.x / BY);
    float reg_C; 
    float *C_start;
    if (coeff_y < dim[dim_C_A[0]] && coeff_x < dim[dim_C_B[0]]){
        idx[dim_C_A[0]] = coeff_y;
        idx[dim_C_B[0]] = coeff_x;

        C_start = C + idx[dim_C_A[0]];
        for (int i=1; i<nd_C_A; ++i){
            idx[dim_C_A[i]] = blockIdx.y / (blocks_y * acc_dims_C_A[i-1]);
            C_start += idx[dim_C_A[i]] * acc_dims_C_A[i];
        }

        C_start += idx[dim_C_B[0]] * acc_dims_C_B[0] * acc_dims_C_A[nd_C_A];
        for (int i=1; i<nd_C_B; ++i){
            idx[dim_C_B[i]] = blockIdx.x / (blocks_x * acc_dims_C_B[i-1]);
            C_start += idx[dim_C_B[i]] * acc_dims_C_B[i] * acc_dims_C_A[nd_C_A];
        }
        reg_C = *C_start; 
    }    
    // A (a e b f)
    //load A from global memory to shared memory
    //                   + a                                               + b * a_dims * e_dims                     + (e) 0 * a_dims  + (f) 0 * a_dims * e_dims * b_dims
    //var/folders/6_/_434ff9j2cz1f0psvgwdf9700000gn/T/TemporaryItems/NSIRD_screencaptureui_XG5uhF/Screen\ Shot\ 2022-01-06\ at\ 2.46.30\ PM.png /float *A_start = A + (blockIdx.y % blocks_y * BY + threadIdx.x % BY) + blockIdx.y / blocks_y * a_dims * e_dims + 0 * a_dims      + threadIdx.x / BY * a_dims * e_dims * b_dims;
    //int A_base = A + (blockIdx.y % blocks_y * BY + threadIdx.x % BY) + blockIdx.y / blocks_y * a_dims * e_dims;   // external indices
    //float *A_start = A_base + 0 * a_dims      + (threadIdx.x / BY+0) * a_dims * e_dims * b_dims;    // internal indices of A
    float *A_start = A;
    int A_base = 0;
    if (dim_A[0]==dim_C_A[0]){
        // the first dim of A is an external dim. A1 case
        A_start += idx[dim_A[0]];
        for (int i=1; i<nd_A; ++i){
            // the dim belongs to external dims
            if ( (1<<dim_A[i]) & Mask ) A_base += idx[dim_A[i]]*acc_dims_A[i];
            // the dim belongs to internal dims
            else {
                int d = threadIdx.x / BY;
                if (d < BK){
                    // <(less than) internal dims
                    A_start += d * acc_dims_A[i];
                }
            }  
        }
    }else{
        // the first dim of A is a contraction dim. A2 case
        if (threadIdx.x / BK < BY){
            A_start += threadIdx.x % BK;
            for (int i=1; i<nd_A; ++i){
                // all the remaining dims are external dims
                A_base += idx[dim_A[i]] * acc_dims_A[i];
            }
        }
    }
    A_start += A_base;
    if ( threadIdx.x < BY*BK ){
        *(sh_A + threadIdx.x) = *(A_start);
    }
    
    // B (f c e d)
    //load B from global memory to shared memory
    //                 + c * f_dims                                               +  d * f_dims * c_dims *e_dims                      + (e) 0 * f_dims * c_dims    + (f) 0
    //float *B_start = B + (blockIdx.x % blocks_x * BX + threadIdx.x / BY) * f_dims + (blockIdx.x / blocks_x) * f_dims * c_dims *e_dims + 0 * f_dims * c_dims        + threadIdx.x % BK;
    //int B_base = B + (blockIdx.x % blocks_x * BX + threadIdx.x / BY) * f_dims + (blockIdx.x / blocks_x) * f_dims * c_dims *e_dims;   // external indices of B
    //float *B_start = B_base + 0 * f_dims * c_dims        + threadIdx.x % BK * 1;     // internal indices of B
    
    float *B_start = B;
    int B_base = 0;
    if (dim_B[0]!=dim_C_B[0]){
        // the first dim of B is a contraction dim. B1 case
        // in this case, we have select it as the first iteration contraction dim previously
        if (threadIdx.x / BK < BX){
            B_start += threadIdx.x % BK;
            for (int i=1; i<nd_B; ++i){
                // all the remaining dims are external dims
                B_base += idx[dim_B[i]]*acc_dims_B[i];
            }
        }
    }else{
        // the first dim of B is an external dim. B2 case
        B_start += idx[dim_B[0]];
        for (int i=1; i<nd_B; ++i){
            // the dim belongs to external dims
            if ( (1<<dim_B[i]) & Mask ) B_base += idx[dim_B[i]]*acc_dims_B[i];
            else{
                // the dim belongs to internal dims
                int d = threadIdx.x / BX;
                if (d < BK){
                    B_start += d * acc_dims_B[i];
                }
            } 
        }
    }
    
    B_start += B_base;
    if (threadIdx.x < BX*BK){
        *(sh_B + threadIdx.x) = *(B_start);
    }
    
    // compute the number of contraction dims
    int prod = dim[dim_A[inter_A[0]]];
    
    // shared_mem: (BY*BK + BX*BK) * 2
    //        0    ~ (BY-1)*BK: A
    // (BY-1)*BK+1 ~ (BY-1)*BK+(BX-1)*BK: B
    int double_buffer = 0;
    float reg_A,reg_B;
    
    int A_offset;
	int B_offset;
    for (int i=0; i<prod; i += BK){
        __syncthreads();
        if (dim_C_A[0]==dim_A[0] && dim_B[0]==inter_B[0]){
            A_offset = double_buffer + threadIdx.x%BY;
		    B_offset = double_buffer + (threadIdx.x/BY) * BK;
        }else if (dim_C_A[0]==dim_A[0]){
            A_offset = double_buffer + threadIdx.x%BY;
		    B_offset = double_buffer + threadIdx.x/BY;
        }else if (dim_B[0]==inter_B[0]){
            A_offset = double_buffer + (threadIdx.x%BY) * BK;
		    B_offset = double_buffer + (threadIdx.x/BY) * BK;
        }else{
            A_offset = double_buffer + (threadIdx.x%BY) * BK;
		    B_offset = double_buffer + threadIdx.x/BY;
        }
        for (int k=0; k<BK; ++k){
            // read A tile from shared memory to registers
            reg_A = sh_A[A_offset];
            // read B tile from shared memory to registers
            reg_B = sh_B[B_offset];
            // perform FMA
            reg_C = fma(reg_A, reg_B, reg_C);

            A_offset += BY;
			B_offset += 1;
        }
        // BX*BK+BY*BK is power of 2
        double_buffer ^= (BX*BK+BY*BK);
        if (i+BK < prod){
            if (dim_C_A[0]==dim_A[0]){
                // external indices
                A_start = A + A_base + (threadIdx.x/BY + i + BK) % prod * acc_dims_A[inter_A[0]];
            }else{
                // internal indices of A. A2 case
                A_start = A + A_base + (threadIdx.x%BK + i + BK) % prod * acc_dims_A[inter_A[0]]; 
            }
            if ( threadIdx.x < BY*BK ){
                *(sh_A + double_buffer + threadIdx.x) = *(A_start);
            }
            if (dim_B[0]==inter_B[0]){
                // internal indices of B. B1 case
                B_start = B + B_base + (threadIdx.x%BK + i + BK) % prod * acc_dims_B[0];  
            }else{
                // external indices
                B_start = B + B_base + (threadIdx.x/BX + i + BK) % prod * acc_dims_B[inter_B[0]];
            }
            if (threadIdx.x < BX*BK){
                *(sh_B + double_buffer + threadIdx.x) = *(B_start);
            }
        }
    }
    // write C tile from register to global memory
    *C_start = reg_C;
    
}
void run_cuTC(Tensor &AA, Tensor &BB, Tensor &CC){
    std::vector<int> modeC(CC.get_Mode());
    std::vector<int> modeA(AA.get_Mode());
    std::vector<int> modeB(BB.get_Mode());
    for (auto &it:modeC){
        it = it-'a';
    } 
    for (auto &it:modeB){
        it = it-'a';
    } 
    for (auto &it:modeA){
        it = it-'a';
    } 
    std::vector<int64_t> extentC(CC.get_Ext());
    //for (auto mode : modeC)
    //    extentC.push_back(extent[mode]);
    std::vector<int64_t> extentA(AA.get_Ext());
    //for (auto mode : modeA)
    //    extentA.push_back(extent[mode]);
    std::vector<int64_t> extentB(BB.get_Ext());
    //for (auto mode : modeB)
    //    extentB.push_back(extent[mode]);

    std::unordered_map<int, int64_t> extent;
    for (int i=0; i<modeC.size(); ++i){
        if (!extent.count(modeC[i])){
            extent[modeC[i]] = extentC[i]; 
        }else if(extent[modeC[i]]!=extentC[i]){
            std::cout<<"C extent error!"<<std::endl;
            exit(1);
        }
    }
    for (int i=0; i<modeA.size(); ++i){
        if (!extent.count(modeA[i])){
            extent[modeA[i]] = extentA[i]; 
        }else if(extent[modeA[i]]!=extentA[i]){
            std::cout<<"A extent error!"<<std::endl;
            exit(1);
        }
    }
    for (int i=0; i<modeB.size(); ++i){
        if (!extent.count(modeB[i])){
            extent[modeB[i]] = extentB[i]; 
        }else if(extent[modeB[i]]!=extentB[i]){
            std::cout<<"B extent error!"<<std::endl;
            exit(1);
        }
    }
    int64_t *dim = (int64_t *)malloc(extent.size()*sizeof(int64_t));
    for (auto it:extent){
        dim[it.first] = it.second;
    }
   
    size_t elementsA = 1;
    for (auto num : extentA)
        elementsA *= num;
    size_t elementsB = 1;
    for (auto num : extentB)
        elementsB *= num;
    size_t elementsC = 1;
    for (auto num : extentC)
        elementsC *= num;

    float *A = (float *) malloc(sizeof(float) * elementsA);
    float *B = (float *) malloc(sizeof(float) * elementsB);
    float *C = (float *) malloc(sizeof(float) * elementsC);

    //if (A == NULL || B == NULL || C == NULL)
    //{
    //    printf("Error: Host allocation of A or C.\n");
    //    return -1;
    //}

    /*******************
     * Initialize data
     *******************/

    for (int64_t i = 0; i < elementsA; i++)
        A[i] = AA.data[i];
    for (int64_t i = 0; i < elementsB; i++)
        B[i] = BB.data[i];
    for (int64_t i = 0; i < elementsC; i++)
        C[i] = CC.data[i];

    int32_t *dim_C = (int32_t *)malloc(modeC.size()*sizeof(int32_t));
    for (int i=0; i<modeC.size(); ++i){
        dim_C[i] = modeC[i];
    }
    tensorContraction_host(A, modeA.data(), modeA.size(),
                           B, modeB.data(), modeB.size(),
                           C, dim_C, modeC.size(),
                           dim, extent.size());
}
void tensorContraction_host(float A[], int dim_A[], int nd_A, \
                    float B[], int dim_B[], int nd_B, \
                    float C[], int32_t dim_C[], int nd_C, \
                    int64_t dim[], int nd){
    // A: {dim[dim_A[0]], dim[dim_A[1]], dim[dim_A[2]], ...}
    // B: {dim[dim_B[0]], dim[dim_B[1]], dim[dim_B[2]], ...}
    // C: {dim[dim_C[0]], dim[dim_C[1]], dim[dim_C[2]], ...}
    int kernel_choice = (nd_A + nd_B - nd_C)/2;
    int mask = 0;
    int size_C = 1;
    for (int i=0; i<nd_C; ++i){
        mask |= (1<<dim_C[i]);
        size_C *= dim_C[i];
    }
    int bit_A = 0;
    int size_A = 1;
    for (int i = 0; i<nd_A; i++){
        bit_A |= (1<<dim_A[i]);
        size_A *= dim_A[i];
    }
    int bit_B = 0;
    int size_B = 1;
    for (int i = 0; i<nd_B; i++){
        bit_B |= (1<<dim_B[i]);
        size_B *= dim_B[i];
    }
    
    int *acc_dims_A_host = (int *) malloc(nd_A * sizeof(int));
    acc_dims_A_host[0] = 1;
    int *inter_A_host = (int *) malloc(kernel_choice * sizeof(int));;
    int k = 0;
    if (!( (mask & bit_A) & (1<<dim_A[0]) )) inter_A_host[k++] = 0;  
    for (int i = 1; i<nd_A; i++){
        if ( !( (mask & bit_A) & (1<<dim_A[i]) ) ) inter_A_host[k++] = i; 
        acc_dims_A_host[i] = acc_dims_A_host[i-1] * dim[dim_A[i-1]];
    }
    
    int *acc_dims_B_host = (int *) malloc(nd_B * sizeof(int));
    acc_dims_B_host[0] = 1;
    int *inter_B_host = (int *) malloc(kernel_choice * sizeof(int));;
    k = 0;
    if (!( (mask & bit_B) & (1<<dim_B[0]) )) inter_B_host[k++] = 0;  
    for (int i = 1; i<nd_B; i++){
        if ( !( (mask & bit_B) & (1<<dim_B[i]) ) ) inter_B_host[k++] = i; 
        acc_dims_B_host[i] = acc_dims_B_host[i-1] * dim[dim_B[i-1]];
    }
    int nd_C_A = nd_A-kernel_choice;
    int nd_C_B = nd_B-kernel_choice;
    int *acc_dims_C_A_host = (int *) malloc((nd_C_A+1) * sizeof(int));
    int *acc_dims_C_B_host = (int *) malloc((nd_C_B+1) * sizeof(int));
    acc_dims_C_A_host[0] = 1;
    acc_dims_C_B_host[0] = 1;
    int *dim_C_A_host = (int *) malloc(nd_C_A * sizeof(int));
    int *dim_C_B_host = (int *) malloc(nd_C_B * sizeof(int));
    int k1 = 0, k2 = 0;
    
    for (int i = 0; i<nd_A; i++){
        if ( mask & (1<<dim_A[i]) ){
            dim_C_A_host[k1++] = dim_A[i];
        }
    }
    for (int i = 1; i<=nd_C_A; i++){
        acc_dims_C_A_host[i] = acc_dims_C_A_host[i-1] * dim[dim_C_A_host[i-1]];
    }
    
    for (int i = 0; i<nd_B; i++){
        if ( mask & (1<<dim_B[i]) ){
            dim_C_B_host[k2++] = dim_B[i];
        }
    }
    for (int i = 1; i<=nd_C_B; i++){
        acc_dims_C_B_host[i] = acc_dims_C_B_host[i-1] * dim[dim_C_B_host[i-1]];
    }
    
    dim3 grid_size(0,0,1);
    int BX = 24;
    int blocks_x = (dim[dim_C_B_host[0]] + BX - 1) / BX;
    int prod = 1;
    for (int i=1; i<nd_C_B; ++i){
        prod *= dim[dim_C_B_host[i]];
    }
    grid_size.x = prod * blocks_x;
    int BY = 16;
    int blocks_y = (dim[dim_C_A_host[0]] + BY - 1) / BY;
    prod = 1;
    for (int i=1; i<nd_C_A; ++i){
        prod *= dim[dim_C_A_host[i]];
    }
    grid_size.y = prod * blocks_y;

    int BK = 12;
    dim3 block_size(BX*BY,1,1);
    int shared_mem_size = sizeof(float)*2*(BX*BK+BY*BK);
    
    // transfer data to the device 
    // tensor A
    float *A_device;
    ErrChk(cudaMalloc((void**)&A_device, size_A*sizeof(float)));
    ErrChk(cudaMemcpy(A_device, A, size_A*sizeof(float), cudaMemcpyHostToDevice));
    
    int *dim_A_device;
    ErrChk(cudaMalloc((void**)&dim_A_device, nd_A*sizeof(int)));
    ErrChk(cudaMemcpy(dim_A_device, dim_A, nd_A*sizeof(int), cudaMemcpyHostToDevice));

    int *inter_A_device;
    ErrChk(cudaMalloc((void**)&inter_A_device, kernel_choice*sizeof(int)));
    ErrChk(cudaMemcpy(inter_A_device, inter_A_host, kernel_choice*sizeof(int), cudaMemcpyHostToDevice));

    int *acc_dims_A_device;
    ErrChk(cudaMalloc((void**)&acc_dims_A_device, nd_A*sizeof(int)));
    ErrChk(cudaMemcpy(acc_dims_A_device, acc_dims_A_host, nd_A*sizeof(int), cudaMemcpyHostToDevice));
    
    // tensor B
    float *B_device;
    ErrChk(cudaMalloc((void**)&B_device, size_B*sizeof(float)));
    ErrChk(cudaMemcpy(B_device, B, size_B*sizeof(float), cudaMemcpyHostToDevice));

    int *dim_B_device;
    ErrChk(cudaMalloc((void**)&dim_B_device, nd_B*sizeof(int)));
    ErrChk(cudaMemcpy(dim_B_device, dim_B, nd_B*sizeof(int), cudaMemcpyHostToDevice));

    int *inter_B_device;
    ErrChk(cudaMalloc((void**)&inter_B_device, kernel_choice*sizeof(int)));
    ErrChk(cudaMemcpy(inter_B_device, inter_B_host, kernel_choice*sizeof(int), cudaMemcpyHostToDevice));

    int *acc_dims_B_device;
    ErrChk(cudaMalloc((void**)&acc_dims_B_device, nd_B*sizeof(int)));
    ErrChk(cudaMemcpy(acc_dims_B_device, acc_dims_B_host, nd_B*sizeof(int), cudaMemcpyHostToDevice));


    // tensor C
    float *C_device;
    ErrChk(cudaMalloc((void**)&C_device, size_C*sizeof(float)));
    ErrChk(cudaMemcpy(C_device, C, size_C*sizeof(float), cudaMemcpyHostToDevice));

    int *dim_C_A_device;
    ErrChk(cudaMalloc((void**)&dim_C_A_device, nd_C_A*sizeof(int)));
    ErrChk(cudaMemcpy(dim_C_A_device, dim_C_A_host, nd_C_A*sizeof(int), cudaMemcpyHostToDevice));

    int *acc_dims_C_A_device;
    ErrChk(cudaMalloc((void**)&acc_dims_C_A_device, (nd_C_A+1)*sizeof(int)));
    ErrChk(cudaMemcpy(acc_dims_C_A_device, acc_dims_C_A_host, (nd_C_A+1)*sizeof(int), cudaMemcpyHostToDevice));

    int *dim_C_B_device;
    ErrChk(cudaMalloc((void**)&dim_C_B_device, nd_C_B*sizeof(int)));
    ErrChk(cudaMemcpy(dim_C_B_device, dim_C_B_host, nd_C_B*sizeof(int), cudaMemcpyHostToDevice));

    int *acc_dims_C_B_device;
    ErrChk(cudaMalloc((void**)&acc_dims_C_B_device, (nd_C_B+1)*sizeof(int)));
    ErrChk(cudaMemcpy(acc_dims_C_B_device, acc_dims_C_B_host, (nd_C_B+1)*sizeof(int), cudaMemcpyHostToDevice));

    // dim
    int64_t *dim_device;
    ErrChk(cudaMalloc((void**)&dim_device, nd*sizeof(int64_t)));
    ErrChk(cudaMemcpy(dim_device, dim, nd*sizeof(int64_t), cudaMemcpyHostToDevice));

    // call the kernel
    if (kernel_choice==1){
        // the contraction-dim is 1
        tc_1dim_64_8X8<<<grid_size,block_size,shared_mem_size>>>(A_device, dim_A_device, inter_A_device, acc_dims_A_device, nd_A, \
                                                                 B_device, dim_B_device, inter_B_device, acc_dims_B_device, nd_B, \
                                                                 C_device, dim_C_A_device, acc_dims_C_A_device, nd_C_A, dim_C_B_device, acc_dims_C_B_device, nd_C_B, mask, \
                                                                 dim_device, nd);
    }else{
        // the contraction-dim is 2
        tc_2dims_64_8X8<<<grid_size,block_size,shared_mem_size>>>(A_device, dim_A_device, inter_A_device, acc_dims_A_device, nd_A, \
                                                                  B_device, dim_B_device, inter_B_device, acc_dims_B_device, nd_B, \
                                                                  C_device, dim_C_A_device, acc_dims_C_A_device, nd_C_A, dim_C_B_device, acc_dims_C_B_device, nd_C_B, mask, \
                                                                  dim_device, nd);
    }
    /*
    // tensor C_final
    float *C_final_device;
    ErrChk(cudaMalloc((void**)&C_final_device, size_C*sizeof(float)));

    // permutation
    typedef float floatTypeA;
    typedef float floatTypeB;
    typedef float floatTypeC;
    typedef float floatTypeCompute;

    cudaDataType_t typeA = CUDA_R_32F;
    cudaDataType_t typeB = CUDA_R_32F;
    cudaDataType_t typeC = CUDA_R_32F;
    cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_32F;
    floatTypeCompute alpha = (floatTypeCompute)1.1f;
    floatTypeCompute beta  = (floatTypeCompute)0.f;

    cutensorHandle_t handle;
    HANDLE_ERROR(cutensorInit(&handle));
    cutensorTensorDescriptor_t descC_final;
    int64_t *extent_C_final = (int64_t *) malloc(nd_C * sizeof(int64_t));
    for (int i = 0; i<nd_C; i++){
        extent_C_final[i] = dim[dim_C[i]];
    }
    HANDLE_ERROR(cutensorInitTensorDescriptor( &handle,
                 &descC_final,
                 nd_C,
                 extent_C_final,
                 NULL,
                 typeC, CUTENSOR_OP_IDENTITY));
    
    int64_t *extent_C_device = (int64_t *) malloc(nd_C * sizeof(int64_t));
    int32_t *dim_C_device = (int32_t *) malloc(nd_C * sizeof(int32_t));
    int k_C = 0;
    for (int i = 0; i<nd_C_A; i++){
        extent_C_device[k_C] = dim[dim_C_A_host[i]];
        dim_C_device[k_C++] = dim_C_A_host[i];
    }
    
    for (int i = 0; i<nd_C_B; i++){
        extent_C_device[k_C] = dim[dim_C_B_host[i]];
        dim_C_device[k_C++] = dim_C_B_host[i];
    }
    cutensorTensorDescriptor_t descC_device;
    HANDLE_ERROR(cutensorInitTensorDescriptor( &handle,
                 &descC_device,
                 nd_C,
                 extent_C_device,
                 NULL,
                 typeC, CUTENSOR_OP_IDENTITY));
    cutensorPermutation(&handle, (void*) &alpha, 
                        C_device, &descC_device, dim_C_device, 
                        C_final_device, &descC_final, dim_C, typeC, 0);

    ErrChk(cudaMemcpy(C, C_final_device, size_C*sizeof(float), cudaMemcpyDeviceToHost));
   */ 
    // varify the correctness
    // in previous level function

    free(acc_dims_A_host);
	free(inter_A_host);
	free(acc_dims_B_host);
	free(inter_B_host);
    
    free(acc_dims_C_A_host);
    free(acc_dims_C_B_host);
	free(dim_C_A_host);
    free(dim_C_B_host);

    /*
    ErrChk(cudaFree(A_device));
    ErrChk(cudaFree(dim_A_device));
    ErrChk(cudaFree(inter_A_device));
    ErrChk(cudaFree(acc_dims_A_device));

    ErrChk(cudaFree(B_device));
    ErrChk(cudaFree(dim_B_device));
    ErrChk(cudaFree(inter_B_device));
    ErrChk(cudaFree(acc_dims_B_device));

    ErrChk(cudaFree(C_device));
    ErrChk(cudaFree(dim_C_A_device));
    ErrChk(cudaFree(acc_dims_C_A_device));
    ErrChk(cudaFree(dim_C_B_device));
    ErrChk(cudaFree(acc_dims_C_B_device));
*/
}
/*
__global__ 
void tensor_contraction_2contraction_Dim_64_8x8(float* A[], int dim_A[], int nd_A, \
                    float* B[], int dim_B[], int nd_B, \
                    float* C[], int dim_C[], int nd_C, \
                    int dim[], int nd, \
                    int dim_intra[], int nd_intra){
    // A: {dim[dim_A[0]], dim[dim_A[1]], dim[dim_A[2]], ...}
    // B: {dim[dim_B[0]], dim[dim_B[1]], dim[dim_B[2]], ...}
    // C: {dim[dim_C[0]], dim[dim_C[1]], dim[dim_C[2]], ...}
    int BY = 8, BX = 8, BK = 8;
    extern __shared__ float sh[]; 
    // shared_mem: BY*BK + BX*BK | BY*BK + BX*BK
    float *sh_A = sh;
	float *sh_B = sh + (BY*BK);  // maybe needs to modified
    
    // C (a,b  X  c,d)
    // horizontal
    int blocks_x = (c_dims + BX - 1)/BX;
    //int block_base_x = blockIDx.x / blocks_x * c_dims + blockIdx.x % blocks_x * BX; // <= N
    
    // vertical
    int blocks_y = (a_dims + BY - 1)/BY;
	//int block_base_y = blocIDx.y / blocks_y * a_dims + blockIdx.y % blocks_y * BY; // <= M                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                lockIdx.x * BY;
    
    // Load C from global memory to register file
    // Here, blockDim.x = BX * BY
    //                 + a                                               + b * a_dims                     + c * a_dims * b_dims                                               + d * a_dims * b_dims * c_dims
	float *C_start = C + (blockIdx.y % blocks_y * BY + threadIdx.x % BY) + blockIdx.y / blocks_y * a_dims + (blockIdx.x % blocks_x * BX + threadIdx.x / BY) * a_dims * b_dims + (blockIdx.x / blocks_x) * a_dims * b_dims * c_dims ;
	reg_C = *C_start; 
    
    // A (a e b f)
    //load A from global memory to shared memory
    //                   + a                                               + b * a_dims * e_dims                     + (e) 0 * a_dims  + (f) 0 * a_dims * e_dims * b_dims
    //float *A_start = A + (blockIdx.y % blocks_y * BY + threadIdx.x % BY) + blockIdx.y / blocks_y * a_dims * e_dims + 0 * a_dims      + threadIdx.x / BY * a_dims * e_dims * b_dims;
    int A_base = A + (blockIdx.y % blocks_y * BY + threadIdx.x % BY) + blockIdx.y / blocks_y * a_dims * e_dims;   // external indices
    float *A_start = A_base + 0 * a_dims      + (threadIdx.x / BY+0) * a_dims * e_dims * b_dims;    // internal indices of A
    *(sh_A + threadIdx.x) = *(A_start);
    // B (f c e d)
    //load B from global memory to shared memory
    //                 + c * f_dims                                               +  d * f_dims * c_dims *e_dims                      + (e) 0 * f_dims * c_dims    + (f) 0
    //float *B_start = B + (blockIdx.x % blocks_x * BX + threadIdx.x / BY) * f_dims + (blockIdx.x / blocks_x) * f_dims * c_dims *e_dims + 0 * f_dims * c_dims        + threadIdx.x % BK;
    int B_base = B + (blockIdx.x % blocks_x * BX + threadIdx.x / BY) * f_dims + (blockIdx.x / blocks_x) * f_dims * c_dims *e_dims;   // external indices of B
    float *B_start = B_base + 0 * f_dims * c_dims        + threadIdx.x % BK * 1;     // internal indices of B
    *(sh_B + threadIdx.x) = *(B_start);
    // shared_mem: (BY*BK + BX*BK) * 2
    //        0    ~ (BY-1)*BK: A
    // (BY-1)*BK+1 ~ (BY-1)*BK+(BX-1)*BK: B
    int double_buffer = 0; 
    for (int i=0; i<e_dims * f_dims; i += BK){
        int A_offset = double_buffer + (threadIdx.x%BY);
		int B_offset = double_buffer + (threadIdx.x/BY)*BK;
            
        for (int k=0; k<BK; ++k){
            // read A tile from shared memory to registers
            reg_A = sh_A[A_offset];
            // read B tile from shared memory to registers
            reg_B = sh_B[B_offset];
            // perform FMA
            reg_C = fma(reg_A, reg_B, reg_C);

            A_offset += BY;
			B_offset += 1;
        }
        // BX*BK+BY*BK = 8*8+8*8 = 128
        double_buffer ^= (BX*BK+BY*BK);
        if (i+BK < e_dims * f_dims){
            // internal indices of A
            f_A = (threadIdx.x/BY + i + BK) % f_dims;
            e_A = (threadIdx.x/BY + i + BK) / f_dims;
            //               + e * a_dims + f * a_dims * e_dims * b_dims
            A_start = A_base + e * a_dims + f * a_dims * e_dims * b_dims; 
            *(sh_A + double_buffer + threadIdx.x) = *(A_start);

            // internal indices of B 
            f_B = (threadIdx.x%BK + i + BK) % f_dims;
            e_B = (threadIdx.x%BK + i + BK) / f_dims; 
            //               + e * f_dims * c_dims + f
            B_start = B_base + e * f_dims * c_dims + f; 
            *(sh_B + double_buffer + threadIdx.x) = *(B_start);
        }
    }
    // write C tile from register to global memory
    *C_start = reg_C;
}
*/
