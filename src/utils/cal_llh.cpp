using namespace std;

inline double llh_cal_approx(const double& evidence, const int& status, const double& m){
  double prob_p = evidence > 0 ? 1./(1. - evidence) : (1./(evidence + 1.)/m);
  double prob_n = evidence < 0 ? (1./(evidence + 1.)) : (1./(1. - evidence)/m);
  double prob_r = evidence > 0 ? (1./evidence) : (-1./evidence);
  double normalizer = prob_n + prob_p + prob_r;
  if(evidence == 0){
    switch(status){
      case -1:
      return -log(m);
      case 1:
      return -log(m);
      case 0:
      return 0;
    }
  }
  switch(status){
    case -1:
    return log(prob_n / normalizer);
    case 1:
    return log(prob_p / normalizer);
    case 0:
    return log(prob_r / normalizer);
  }
}

double cal_llh(const sparse_csr_weighted& proposal_csr, const network_in_device& nw_host,
  const n_nodes& proposal_node){

  double llh = 0;
  const n_nodes& num_nodes = *nw_host.csr_info.number_of_nodes;
  const uint8_t& t_length = *nw_host.nw_info.time_length;

  // borrow the memory in nw_host
  for(int t = 0; t < t_length; t++){
    for(int node = 0; node < num_nodes; node++){
      if(t == 0){
        // assume initialized
        continue;
      }
      // node's type preassumed regular
      // otherwise no need to update
      const int& linked_node_start = nw_host.csr_info.row_ptr[node];
      const int& linked_node_end = nw_host.csr_info.row_ptr[node + 1];

      double& current_evidence = nw_host.sim_ptr.evidence[num_nodes * t + node];
      double& prev_evidence = nw_host.sim_ptr.evidence[num_nodes * (t - 1) + node];
      int node_status = nw_host.sim_ptr.activated_positive[num_nodes * t + node]
        - nw_host.sim_ptr.activated_negative[num_nodes * t + node];

      if(node == proposal_node){
        current_evidence = prev_evidence * proposal_csr.confidence[node];
        for(int node2 = linked_node_start; node2 < linked_node_end; node2++){
          const int& node2_index = nw_host.csr_info.col_index[node2];
          current_evidence += nw_host.sim_ptr.evidence[num_nodes * (t - 1) + node2_index]
            * proposal_csr.influence[node2];
        }
        llh += llh_cal_approx(current_evidence, node_status, 100);
      }else{
        current_evidence = prev_evidence * nw_host.csr_info.confidence[node];
        for(int node2 = linked_node_start; node2 < linked_node_end; node2++){
          const int& node2_index = nw_host.csr_info.col_index[node2];
          current_evidence += nw_host.sim_ptr.evidence[num_nodes * (t - 1) + node2_index]
            * nw_host.csr_info.influence[node2];
        }
        llh += llh_cal_approx(current_evidence, node_status, 100);
      }
    }
  }

  return llh;
}

void accept_proposal(const sparse_csr_weighted& proposal_csr, const network_in_device& nw_host,
  const n_nodes& proposal_node){
  const int& linked_node_start = nw_host.csr_info.row_ptr[proposal_node];
  const int& linked_node_end = nw_host.csr_info.row_ptr[proposal_node + 1];

  nw_host.csr_info.confidence[proposal_node] = proposal_csr.confidence[proposal_node];

  for(int node = linked_node_start; node < linked_node_end; node++){
    nw_host.csr_info.influence[node] = proposal_csr.influence[node];
  }
  return;
}
