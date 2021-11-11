#ifndef TENSORS_H
#define TENSORS_H 
#include "io.h"
class Tensor{
    public:
        Tensor(vector<int>& mode, unordered_map<int,int64_t>& extents);
    private:
        vector<TENSOR_TYPE> data; 
        vector<int> mode; // dimension ID 
        unordered_map<int,int64_t> extents; // dimentsion sizes
};
#endif /* TENSORS_H */
