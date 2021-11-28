#ifndef TENSORS_H
#define TENSORS_H 
#include "io.h"
#include "common.h"
class Tensor{
    public:
        Tensor(vector<int>& mod, unordered_map<int,int64_t>& exts, bool initiallize);
        Tensor(vector<int>& mod, unordered_map<int,int64_t>& exts);
        size_t sz = 1;
        vector<int>& get_Mode();
        vector<int>& get_Ext();
    private:
        vector<TENSOR_TYPE> data; 
        vector<int> mode; // dimension ID 
        vector<int> extents; // dimentsion sizes

        void genTensor();
        void iniTensor();
};
#endif /* TENSORS_H */
