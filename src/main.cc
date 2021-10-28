#include "../include/main.h"

DEFINE_string(bench, "./data/tccg_bench", "The name of benchmarks.");
DEFINE_int32(n, 1, "The number of running times.");
//DEFINE_string(type, "b","(b)cuTENSOR OR (c)cuTC");

int main(int argc, char *argv[])
{
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  VLOG(2)<<"Loading bench config....";
  std::shared_ptr<Tensor> tensors = std::make_shared<Tensor>(FLAGS_bench);  
 
  


  return 0;

}
