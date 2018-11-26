#include "../src/include.h"

using namespace std;
int main(){
  sbm_parameters test_sbm_pars = gen_sbm_pars(10, 1, 1, 1, 0.1);
  sparse_csr_weighted test_csr_info = generate_sparse(test_sbm_pars);
  network_info test_info = init_info(test_csr_info, 0.1, 2, 0.8);

  network_in_device test_nw_info = init_simulation(test_csr_info, test_info);
  string test_csr_filename = "test_csr.data";
  string test_info_filename = "test_info.data";

  save_network(test_nw_info, test_info_filename, test_csr_filename);
  return 0;
}