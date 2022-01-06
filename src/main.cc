#include "../include/main.h"

DEFINE_string(bench, "../data/tccg_bench.in", "The name of benchmarks.");
DEFINE_int32(n, 1, "The number of running times.");
//DEFINE_string(type, "b","(b)cuTENSOR OR (c)cuTC");

int main(int argc, char *argv[])
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    VLOG(2)<<"Loading bench config....";
    std::shared_ptr<Config> cfg = std::make_shared<Config>(FLAGS_bench);  
    //cfg->print();
  
    Tensor A(cfg->get_Mode_a(),cfg->get_Extents(),true);  
    Tensor B(cfg->get_Mode_b(),cfg->get_Extents(),true);  
    Tensor C(cfg->get_Mode_c(),cfg->get_Extents());  
  
    //run_cuTensor(A,B,C);
    run_cuTC(A,B,C);

    return 0;

}
