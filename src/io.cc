#include "../include/io.h"

Config::Config(const std::string bench_name):conFil(bench_name.substr(bench_name.find_last_of("/")+1,-1)){
    parseTensor(conFil);
    set_a();
    set_b();
}
std::unordered_map<int,int64_t>& Config::get_Extents(){
    return extents;
}    
void Config::set_a(){
    for (auto md:mode_a){
        ext_a.push_back(extents[md]);
    }
}
void Config::set_b(){
    for (auto md:mode_b){
        ext_b.push_back(extents[md]);
    }
}
std::vector<int>& Config::get_Mode_a(){
    return mode_a;
}    
std::vector<int>& Config::get_Mode_b(){
    return mode_b;
}    
std::vector<int>& Config::get_Mode_c(){
    return mode_c;
}    
std::vector<int>& Config::get_Ext_a(){
    return ext_a;
}    
std::vector<int>& Config::get_Ext_b(){
    return ext_b;
}    
std::vector<int>& Config::get_Ext_c(){
    return ext_c;
}    
void Config::parseTensor(const std::string& s){
    std::cout<<"s:"<<s<<std::endl;
    std::fstream file;
    file.open(s,std::ios::in);
    std::string abc;
    std::getline(file,abc);
        std::cout<<"abc:"<<abc<<std::endl;
        std::size_t c_start = abc.find('[',0);
        std::size_t c_end = abc.find(']',0);
        std::size_t contraction_start = abc.find('(',0);
        std::size_t contraction_end = abc.find(')',0);
        std::size_t a_start = abc.find('[',contraction_end);
        std::size_t a_end = abc.find(']',contraction_end);
        std::size_t b_start = abc.find('[',a_end);
        std::size_t b_end = abc.find(']',a_end);
        char dim;
        std::stringstream a_s(abc.substr(c_start+1,c_end-c_start-1));
        std::string word;
        while (std::getline(a_s,word,',')){
            char ch = word[0];
            if (ch>='a' && ch<='z'){
                dim = ch;
                continue;
            }
            extents[dim] = std::stoi(word); 
            ext_c.push_back(std::stoi(word));
            mode_c.push_back(dim);
        }
        std::stringstream contract_s(abc.substr(contraction_start+1,contraction_end-contraction_start-1));
        while (std::getline(contract_s,word,',')){
            char ch = word[0];
            if (ch>='a' && ch<='z'){
                dim = ch;
                continue;
            }
            extents[dim] = std::stoi(word); 
            mode_ct.push_back(dim);
        }
        for (auto ch:abc.substr(a_start+1,a_end-a_start-1)){
            if (ch>='a' && ch<='z'){
                mode_a.push_back(ch);
            }
        }
        for (auto ch:abc.substr(b_start+1,b_end-b_start-1)){
            if (ch>='a' && ch<='z'){
                mode_b.push_back(ch);
            }
        }
    file.close();
}
void Config::print(){
    std::cout<<"a:"<<std::endl;
    for (auto m:mode_a){
        std::cout<<(char)m<<":"<<extents[m]<<std::endl;
    }
    std::cout<<"b:"<<std::endl;
    for (auto m:mode_b){
        std::cout<<(char)m<<":"<<extents[m]<<std::endl;
    }
    std::cout<<"c:"<<std::endl;
    for (auto m:mode_c){
        std::cout<<(char)m<<":"<<extents[m]<<std::endl;
    }
    std::cout<<"contraction:"<<std::endl;
    for (auto m:mode_ct){
        std::cout<<m<<":"<<extents[m]<<std::endl;
    }
}

