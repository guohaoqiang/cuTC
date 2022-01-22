#include "../include/tensor.h"

Tensor::Tensor(std::vector<int>& mod, std::unordered_map<int,int64_t>& exts, bool initiallize){
    this->mode = mod;
    for (auto &it:this->mode){
        extents.push_back(exts[it]);
        sz *= exts[it];
    }
    data.reserve(sz);
    if (initiallize)
        genTensor();
}

Tensor::Tensor(std::vector<int>& mod, std::unordered_map<int,int64_t>& exts){
    this->mode = mod;
    for (auto &it:this->mode){
        extents.push_back(exts[it]);
        sz *= exts[it];
    }
    data.reserve(sz);
    iniTensor();
}
void Tensor::iniTensor(){
    for (size_t i = 0; i < sz; i++){
        data[i] = (TENSOR_TYPE)0.0;
    }
}
void Tensor::genTensor(){
    for (size_t i = 0; i < sz; i++){
        //data[i] = (((TENSOR_TYPE) rand())/RAND_MAX - 0.5)*100;
        data[i] = 1.0;
    }
}

std::vector<int> Tensor::get_Mode(){
    return this->mode;
}    
std::vector<int64_t> Tensor::get_Ext(){
    return this->extents;
}    
