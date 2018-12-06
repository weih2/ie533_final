// __host__
#ifndef NAIVE_GREEDY
#define NAIVE_GREEDY

using namespace std;

__global__
void cal_obj(simulation_greedy sim_greedy, network_in_device nw_device){
  int new_exp_id = threadIdx.x;
  int num_nodes = *nw_device.csr_info.number_of_nodes;

  if(
    (new_exp_id >= num_nodes)
    ||
    (nw_device.nw_info.nodes_types[new_exp_id] != NODE_TYPE_REGULAR)
  ) return;

  sim_greedy.objective[new_exp_id] = 0;

  for(int i = 0; i < num_nodes; i++){
    sim_greedy.objective[new_exp_id] +=
      (sim_greedy.total_activated_p[new_exp_id * num_nodes + i]
      - sim_greedy.total_activated_n[new_exp_id * num_nodes + i]);
  }
}

// host : give the best next node
int* naive_greedy(network_in_device nw_host, network_in_device nw_device, int n_positive){
  int *best_nodes = new int[n_positive];
  int max_obj;

  const int& num_nodes = *nw_host.csr_info.number_of_nodes;
  const uint8_t& t_length = *nw_host.nw_info.time_length;

  simulation_greedy sim_greedy = init_greedy(nw_host);
  int *objective = new int[num_nodes];
  cudaFree(nw_device.sim_ptr.total_activated_positive);
  cudaFree(nw_device.sim_ptr.total_activated_negative);

  node_type* node_type_p = new node_type;
  *node_type_p = NODE_TYPE_STUBBORN_P;
  node_type* node_type_r = new node_type;
  *node_type_r = NODE_TYPE_REGULAR;

  for(int n_done = 0; n_done < n_positive; n_done++){
    // loop to get results
    max_obj = - num_nodes * t_length - 1;
    for(int node =  0; node < num_nodes; node ++){
      if(nw_host.nw_info.nodes_types[node] != NODE_TYPE_REGULAR){
        // max_obj--;
        continue;
      }
      cudaMemcpy((nw_device.nw_info.nodes_types + node), node_type_p, sizeof(node_type),cudaMemcpyHostToDevice);
      nw_device.sim_ptr.total_activated_positive = sim_greedy.total_activated_p + num_nodes * node;
      nw_device.sim_ptr.total_activated_negative = sim_greedy.total_activated_n + num_nodes * node;
      for(int t = 0; t < t_length; t++){
        device_cal_evidence_global<<<1, 1024>>>(nw_device, t);
        cudaDeviceSynchronize();
      }
      cudaMemcpy((nw_device.nw_info.nodes_types + node), node_type_r, sizeof(node_type),cudaMemcpyHostToDevice);
    }

    // calculate the final results
    cal_obj<<<1, 1024>>>(sim_greedy, nw_device);
    cudaDeviceSynchronize();

    // copy result back
    cudaMemcpy(objective, sim_greedy.objective, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // find optimal node and return
    for(int node = 0; node < num_nodes; node ++){
      if(nw_host.nw_info.nodes_types[node] != NODE_TYPE_REGULAR) continue;
      if(max_obj < objective[node]){
        best_nodes[n_done] = node;
        max_obj = objective[node];
      }
    }
    cout << "zuihoushi: " << max_obj << endl;
    nw_host.nw_info.nodes_types[best_nodes[n_done]] = *node_type_p;
    cudaMemcpy((nw_device.nw_info.nodes_types + best_nodes[n_done]), node_type_p, sizeof(node_type), cudaMemcpyHostToDevice);
  }

  return best_nodes;
}

#endif
