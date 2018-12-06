// device utility to calculate target evidence
__device__
void device_cal_evidence(network_in_device nw_device, int node_id, int t){
  sparse_csr_weighted& csr_info = nw_device.csr_info;
  network_info& nw_info = nw_device.nw_info;
  simulation_single& sim_ptr = nw_device.sim_ptr;

  const n_nodes num_nodes = *csr_info.number_of_nodes;
  double& confidence = csr_info.confidence[node_id];
  const double& p_threshold = *nw_info.p_threshold;
  const double& n_threshold = *nw_info.n_threshold;

  double *current_evidence = sim_ptr.evidence + t * num_nodes + node_id;

  int *p_activated = sim_ptr.activated_positive + t * num_nodes + node_id;
  int *n_activated = sim_ptr.activated_negative + t * num_nodes + node_id;

  int *total_activated_p = sim_ptr.total_activated_positive + node_id;
  int *total_activated_n = sim_ptr.total_activated_negative + node_id;


  if(t == 0){
    if(nw_info.nodes_types[node_id] == NODE_TYPE_STUBBORN_P){
      *current_evidence = 1;
      *p_activated = 1;
      *n_activated = 0;
      *total_activated_p = 1;
      *total_activated_n = 0;
    }

    else if(nw_info.nodes_types[node_id] == NODE_TYPE_STUBBORN_N){
      *current_evidence = -1;
      *p_activated = 0;
      *n_activated = 1;
      *total_activated_p = 0;
      *total_activated_n = 1;
    }

    else{
      *current_evidence = 0;
      *p_activated = 0;
      *n_activated = 0;
      *total_activated_p = 0;
      *total_activated_n = 0;
    }
    return;
  }

  double *prev_evidence = sim_ptr.evidence + (t - 1) * num_nodes + node_id;

  if(nw_info.nodes_types[node_id] != NODE_TYPE_REGULAR){
    *current_evidence = *prev_evidence;
    if(nw_info.nodes_types[node_id] == NODE_TYPE_STUBBORN_N) (*total_activated_n)++;
    if(nw_info.nodes_types[node_id] == NODE_TYPE_STUBBORN_P) (*total_activated_p)++;
    return;
  }

  (*current_evidence) = (*prev_evidence) * confidence;

  int n_linked_nodes = csr_info.row_ptr[node_id + 1] - csr_info.row_ptr[node_id];
  int node_ind;
  for(int node = 0; node < n_linked_nodes; node++){
    node_ind = csr_info.col_index[csr_info.row_ptr[node_id] + node];

    if(
      *(sim_ptr.activated_positive + (t - 1) * num_nodes + node_ind)
      ||
      *(sim_ptr.activated_negative + (t - 1) * num_nodes + node_ind)
    ){
      *current_evidence +=
        (csr_info.influence[csr_info.row_ptr[node_id] + node])
        *
        (*(sim_ptr.evidence + (t - 1) * num_nodes + node_ind));
    }
    // only activated nodes can send evidence
    
  }

  if(*current_evidence > p_threshold){
    *p_activated = 1;
    *n_activated = 0;
    (*total_activated_p) ++;
  }
  else if(*current_evidence < n_threshold){
    *p_activated = 0;
    *n_activated = 1;
    (*total_activated_n) ++;
  }else{
    *p_activated = 0;
    *n_activated = 0;
  }

  return;
}

__global__
void device_cal_evidence_global(network_in_device nw_device, int t){
  // less than 1024 nodes case
  const n_nodes num_nodes = *nw_device.csr_info.number_of_nodes;
  int node_id = threadIdx.x;
  if(node_id < num_nodes){
    device_cal_evidence(nw_device, node_id, t);
  }  
}


simulation_single device_cal_evidence_host(const sparse_csr_weighted &csr_info, const network_info &h_nw_info){
  // we need number of threads equal to the nmber of nodes
  network_in_device nw_device = cp_to_device(csr_info, h_nw_info);

  const int t_length = *h_nw_info.time_length;
  const int n_threads = 1024;
  for(int t = 0; t < t_length; t++){
    // for less than 1024 nodes
    device_cal_evidence_global<<<1, n_threads>>>(nw_device, t);
    cudaDeviceSynchronize();
  }

  return cp_to_host(nw_device);
}
