#ifndef IO_H
#define IO_H 
#include <ifstream>
class Config{
    public:
        Config(const string& bench_name);
        vector<int>& get_Mode_a();
        vector<int>& get_Mode_b();
        vector<int>& get_Mode_c();
        unordered_map<int,int64_t>& get_Extents();
    private:
        void parseTensor(const string& s);
        void print();
        
        string conFil;
        vector<int> mode_a; // dimension ID
        vector<int> mode_b; // dimension ID
        vector<int> mode_c; // dimension ID
        vector<int> mode_ct; // dimension ID
        unordered_map<int,int64_t> extents; // dimentsion sizes

}
#endif /* IO_H */
