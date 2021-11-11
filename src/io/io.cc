#include "io.h"

Config::Config(const string& bench_name):conFil(bench_name){
    parseTensor(conFil);
}

void Config::parseTensor(const string& s){
    std::fstream file;
    file.open(s,std::ios::in);
    
    std::string abc;
    while (std::getline(file,abc)){
        std::size_t c_start = abc.find('[',0);
        std::size_t c_end = abc.find(']',0);
        std::size_t contraction_start = abc.find('(',0);
        std::size_t contraction_end = abc.find(')',0);
        std::size_t a_start = abc.find('[',contraction_end);
        std::size_t a_end = abc.find(']',contraction_end);
        std::size_t b_start = abc.find('[',a_end);
        std::size_t b_end = abc.find(']',a_end);
        char dim;
        for (auto ch:abc.substr(c_start+1,c_end-c_start-1)){
            if (ch==',')    continue;
            if (ch>='a' && ch<='z'){
                dim = ch;
                continue;
            }else{
                extents[dim] = std::stoi(ch); 
                mode_c.push_back(dim);
            }
        }
        for (auto ch:abc.substr(contraction_start+1,contraction_end-contraction_start-1)){
            if (ch>='a' && ch<='z'){
                mode_ct.push_back(dim);
            }
        }
        for (auto ch:abc.substr(a_start+1,a_end-a_start-1)){
            if (ch==',')    continue;
            if (ch>='a' && ch<='z'){
                dim = ch;
                continue;
            }else{
                extents[dim] = std::stoi(ch); 
                mode_a.push_back(dim);
            }
        }
        for (auto ch:abc.substr(b_start+1,b_end-b_start-1)){
            if (ch==',')    continue;
            if (ch>='a' && ch<='z'){
                dim = ch;
                continue;
            }else{
                extents[dim] = std::stoi(ch); 
                mode_b.push_back(dim);
            }
        }
    }
    file.close();
}
void Config::print(){
    std::cout<<"a:"<<std::endl;
    for (auto m:mode_a){
        std::cout<<m<<":"<<extents[m]<<std::endl;
    }
    std::cout<<"b:"<<std::endl;
    for (auto m:mode_b){
        std::cout<<m<<":"<<extents[m]<<std::endl;
    }
    std::cout<<"c:"<<std::endl;
    for (auto m:mode_c){
        std::cout<<m<<":"<<extents[m]<<std::endl;
    }
    std::cout<<"contraction:"<<std::endl;
    for (auto m:mode_ct){
        std::cout<<m<<":"<<extents[m]<<std::endl;
    }
}
