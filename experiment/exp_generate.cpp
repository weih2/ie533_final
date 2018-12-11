#include "../src/include.h"

/*
gen_sbm_pars(const n_nodes& num_nodes, const int& n_communities, const n_nodes& e_leaders,
 const n_nodes& e_n_friends, const double& leader_influence)

init_info(sparse_csr_weighted &csr_info, const double& p_negative,
   const uint8_t& t_length, const double& abs_threshold)
*/


using namespace std;
int main(){
  sbm_parameters exp_sbm_pars = gen_sbm_pars(20, 1, 1, 3, 0.5);
  sparse_csr_weighted exp_csr_info = generate_sparse(exp_sbm_pars);
  network_info exp_nw_info = init_info(exp_csr_info, 0.1, 10, 0.8);

  network_in_device exp_nw_host = init_simulation(exp_csr_info, exp_nw_info);
  string exp_csr_filename;
  string exp_info_filename;

  int n_n_nodes = 5;
  int n_t_length = 5;

  vector<int> n_nodes_vec{20, 40, 60, 80, 100};
  vector<int> t_length_vec{10, 20, 40, 60, 100};

  for(int i = 0; i < n_n_nodes; i++){
    for(int j = 0; j < n_t_length; j++){
      exp_sbm_pars = gen_sbm_pars(n_nodes_vec[i], 1, 1, 6, 0.5);
      exp_csr_info = generate_sparse(exp_sbm_pars);
      exp_nw_info = init_info(exp_csr_info, 0.1, t_length_vec[j], 0.8);

      exp_nw_host = init_simulation(exp_csr_info, exp_nw_info);

      exp_csr_filename = "exp_csr_" + to_string(n_nodes_vec[i]) + "_" + to_string(t_length_vec[j]) + ".data";
      exp_info_filename = "exp_info" + to_string(n_nodes_vec[i]) + "_" + to_string(t_length_vec[j]) + ".data";

      save_network(exp_nw_host, exp_info_filename, exp_csr_filename);
    }
  }

  return 0;
}
