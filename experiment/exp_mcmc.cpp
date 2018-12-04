#include "../src/include.h"

int main(){
  sbm_parameters test_sbm_pars = gen_sbm_pars(100, 1, 1, 5, 0.2);
  adjacency_list sbm_adj_list = generate_sbm(test_sbm_pars);
  influence_list test_influence = generate_influence(sbm_adj_list);
  influence_list test_init_influence = generate_influence(sbm_adj_list);

  sparse_csr_weighted test_csr = to_sparse(sbm_adj_list, test_influence);
  sparse_csr_weighted init_test_csr = to_sparse(sbm_adj_list, test_init_influence);
  network_info test_info = init_info(test_csr, 0.1, 30, 0.8);

  network_in_device test_nw_info = init_simulation(test_csr, test_info);
  generate_influence(test_nw_info, 0.1);

  test_nw_info.csr_info = init_test_csr;
  for(int i = 0; i < 20; i++){
    mcmc(test_nw_info, 100);
    cout << "100 steps done" << endl;
  }
  cout << test_nw_info.csr_info.confidence[0] << endl;

  string test_par_name = "test_true_pars.data";
  string test_par_mcmc_name = "mcmc_result_pars.data";

  save_csr(test_csr, test_par_name);
  save_csr(init_test_csr, test_par_mcmc_name);

  return 0;
}
