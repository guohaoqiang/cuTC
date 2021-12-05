#ifndef TENSORS_H
#define TENSORS_H 
#include "io.h"
#include "common.h"
class Tensor{
    public:
        Tensor(std::vector<int>& mod, std::unordered_map<int,int64_t>& exts, bool initiallize);
        Tensor(std::vector<int>& mod, std::unordered_map<int,int64_t>& exts);
        size_t sz = 1;
        std::vector<int> get_Mode();
        std::vector<int64_t> get_Ext();
        std::vector<TENSOR_TYPE> data; 
    private:
        std::vector<int> mode; // dimension ID 
        std::vector<int64_t> extents; // dimentsion sizes

        void genTensor();
        void iniTensor();
};
#endif /* TENSORS_H */
