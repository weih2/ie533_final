void mcmc(network_in_device& nw_host, const int& n_steps){
  sparse_csr_weighted proposal_csr = init_mcmc(nw_host);
  const int& num_nodes = *nw_host.csr_info.number_of_nodes;
  double old_llh = cal_llh(proposal_csr, nw_host, num_nodes);
  cout << old_llh << endl;
  double new_llh;
  double accept_prob;

  for(int step = 0; step < n_steps; step++){
    random_proposal(proposal_csr, nw_host.csr_info, 10000);
    for(int node = 0; node < num_nodes; node++){
      new_llh = cal_llh(proposal_csr, nw_host, node);
      accept_prob = (new_llh - old_llh);
      accept_prob = (accept_prob > 0? 1:exp(accept_prob));
      if(random_bernoulli(accept_prob)){
        accept_proposal(proposal_csr, nw_host, node);
        old_llh = new_llh;
      }
    }
  }

  return;
}
