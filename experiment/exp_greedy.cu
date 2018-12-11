#include "../src/cuda_include.h"

using namespace std;

int main(){
  char filename_buffer[50];

  string exp_csr_filename;
  string exp_info_filename;
  network_in_device nw_host;
  network_in_device nw_device;

  int n_n_nodes = 5;
  int n_t_length = 5;

  vector<int> n_nodes_vec(0);
  n_nodes_vec.push_back(20);
  n_nodes_vec.push_back(40);
  n_nodes_vec.push_back(60);
  n_nodes_vec.push_back(80);
  n_nodes_vec.push_back(100);

  vector<int> t_length_vec(0);
  t_length_vec.push_back(10);
  t_length_vec.push_back(20);
  t_length_vec.push_back(40);
  t_length_vec.push_back(60);
  t_length_vec.push_back(100);
  // {10, 20, 40, 60, 100};

  for(int i = 0; i < n_n_nodes; i++){
    for(int j = 0; j < n_t_length; j++){
      sprintf(filename_buffer, "exp_csr_%d_%d.data",n_nodes_vec[i],t_length_vec[j]);
      exp_csr_filename = filename_buffer;
      sprintf(filename_buffer, "exp_info%d_%d.data",n_nodes_vec[i],t_length_vec[j]);
      exp_info_filename = filename_buffer;

      nw_host = read_network(exp_info_filename, exp_csr_filename);
      nw_device = cp_to_device(nw_host.csr_info, nw_host.nw_info);

      int *greedy_result = naive_greedy(nw_host, nw_device, 5);
      cout << greedy_result[5] << endl;
      break;
    }
    break;
  }
  return 0;
}