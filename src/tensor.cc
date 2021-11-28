#include "../include/tensor.h"

Tensor::Tensor(vector<int>& mod, unordered_map<int,int64_t>& exts, bool initiallize){
    this->mode = mod;
    for (auto &it:this->mode){
        extents.push_back(exts[it]);
        sz *= exts[it];
    }
    data.reserve(sz);
    if (initiallize)
        genTensor();
}

Tensor::Tensor(vector<int>& mod, unordered_map<int,int64_t>& exts){
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
        data[i] = (float)0.0;
    }
}
void Tensor::genTensor(){
    for (size_t i = 0; i < sz; i++){
        data[i] = (((float) rand())/RAND_MAX - 0.5)*100;
    }
}

vector<int>& Tensor::get_Mode(){
    return this->mode;
}    
vector<int>& Tensor::get_Ext(){
    return this->extents;
}    
