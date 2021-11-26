#ifndef IO_H
#define IO_H 
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <string.h>
#include <unordered_map>
class Config{
    public:
        Config(const std::string bench_name);
        std::vector<int>& get_Mode_a();
        std::vector<int>& get_Mode_b();
        std::vector<int>& get_Mode_c();
        std::unordered_map<char,int64_t>& get_Extents();
        void print();
    private:
        void parseTensor(const std::string& s);
        
        std::string conFil;
        std::vector<char> mode_a; // dimension ID
        std::vector<char> mode_b; // dimension ID
        std::vector<char> mode_c; // dimension ID
        std::vector<char> mode_ct; // dimension ID
        std::unordered_map<char,int64_t> extents; // dimentsion sizes

};
#endif /* IO_H */
