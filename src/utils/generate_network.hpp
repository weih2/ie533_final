#ifndef GEN_NETWORK
#define GEN_NETWORK

adjacency_list generate_sbm(const sbm_parameters& sbm_pars){
  const int& number_of_nodes = sbm_pars.number_of_nodes;
  const vector<double>& class_prob = sbm_pars.class_probability;
  const vector<vector<double>>& link_prob = sbm_pars.link_probability_matrix;

  adjacency_list sbm_adj_list(number_of_nodes);
  // generate an sbm according to the parameters
  vector<int> nodes_block(number_of_nodes);

  for(int node = 0; node < number_of_nodes; node++){
    nodes_block[node] = random_discrete(class_prob);
  }

  for(int node1 = 0; node1 < number_of_nodes; node1++){
    for(int node2 = 0; node2 < number_of_nodes; node2++){
      if(node1 >= node2) continue;

      const double& prob_link = link_prob[nodes_block[node1]][nodes_block[node2]];
      if(random_uniform() < prob_link){
        // a link is added
        sbm_adj_list[node1].push_back(node2);
        // non-directed
        sbm_adj_list[node2].push_back(node1);
      }
    }
  }

  return sbm_adj_list;
}

influence_list generate_influence(const adjacency_list& adj_list){
  // generate a weighted sbm according to the parameters
  const n_nodes num_nodes = adj_list.size();
  influence_list influence(num_nodes);

  int n_adj_nodes;
  vector<double> node_i_weights;
  for(int i = 0; i < num_nodes; i++){
    n_adj_nodes = adj_list[i].size();
    influence[i] = vector<double>(n_adj_nodes + 1);

    // stubornness of node i
    double& node_i_stubborn = influence[i][n_adj_nodes] = random_uniform();
    node_i_weights = random_dirichlet(vector<double>(n_adj_nodes, 10));
    for(int j = 0; j < n_adj_nodes; j++){
      influence[i][j] = (node_i_weights[j]) * (1 - node_i_stubborn);
    }
  }

  return influence;
}

#endif
