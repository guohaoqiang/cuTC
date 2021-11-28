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
        void set_a();
        void set_b();
        std::vector<int>& get_Mode_a();
        std::vector<int>& get_Mode_b();
        std::vector<int>& get_Mode_c();
        std::unordered_map<int,int64_t>& get_Extents();
        std::vector<int>& get_Ext_a();
        std::vector<int>& get_Ext_b();
        std::vector<int>& get_Ext_c();
        void print();
    private:
        void parseTensor(const std::string& s);
        
        std::string conFil;
        std::vector<int> mode_a; // dimension ID
        std::vector<int> mode_b; // dimension ID
        std::vector<int> mode_c; // dimension ID
        
        std::vector<int> ext_a; // dimension ID
        std::vector<int> ext_b; // dimension ID
        std::vector<int> ext_c; // dimension ID
        
        std::vector<int> mode_ct; // dimension ID
        std::unordered_map<int,int64_t> extents; // dimentsion sizes

};
#endif /* IO_H */
