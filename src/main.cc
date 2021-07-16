#include "../include/main.h"

DEFINE_int32(pe_counts, 16, "Total number of PEs. n == r * c");
DEFINE_int32(ar, 4, "Aspect ratio R.");
DEFINE_int32(ac, 4, "Aspect ratio C.");
DEFINE_string(data_name, "./data/demo6", "The name of datasets.");

DEFINE_int32(l1_size, 4, "The size of L1 Cache for all PE cluster (in MB).");
DEFINE_int32(l2_size, 8, "The size of L2 (in MB).");
DEFINE_int32(w, 4, "The sliding window size along columns of adj.");
DEFINE_string(type, "b","(b)gcnax OR (w)wegnn");

int main(int argc, char *argv[])
{
  /*
  optparse::OptionParser parser = optparse::OptionParser();
  parser.add_option("-n", "--pe_counts").type("int").set_default("16").help("Total number of PEs. n == r * c");
  parser.add_option("-r", "--ar").type("int").set_default("1").help("Aspect ratio R.");
  parser.add_option("-c", "--ac").type("int").set_default("1").help("Aspect ratio C.");
  parser.add_option("-d", "--data_name").type("string").set_default("../data/test_data").help("The name of datasets.");
  parser.add_option("-l1", "--l1_size").type("int").set_default("4").help("The size of L1 Cache for all PE cluster (in MB).");
  parser.add_option("-l2", "--l2_size").type("int").set_default("8").help("The size of L2 (in MB).");
  parser.add_option("-t", "--type").type("string").set_default("b").help("(b)gcnax OR (w)wegnn");
  
  optparse::Values& option = parser.parse_args(argc,argv);
  std::shared_ptr<Graph> adj_csr = std::make_shared<Graph>((std::string)option.get("data_name"));  
  */
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  VLOG(2)<<"Loading data....";
  std::shared_ptr<Graph> adj_csr = std::make_shared<Graph>(FLAGS_data_name);  
  VLOG(2)<<"Loading data over....";
  //adj_csr->print_data();
 
   
  std::shared_ptr<Acc> acc = std::make_shared<Acc>(FLAGS_l2_size,FLAGS_l1_size,\
          FLAGS_pe_counts,FLAGS_ar,FLAGS_ac);
  Analysis<char> res(acc,adj_csr,FLAGS_w);
  //res.val();
  res.val_naive1();
  /*
  std::string my_kernel((std::string)option.get("type"));
  if(!my_kernel.compare("w")){
    res.run_wegnn();
  //res.run_w_order_wo_move(3,1,1,1);
  //res.run_w_order_wo_move(1,85,1,1);
  }else{
    res.run_baseline();
  //  res.run_baseline_awbgcn_gcnax(1,2,1,1);
  }
  */
 


  return 0;

}
