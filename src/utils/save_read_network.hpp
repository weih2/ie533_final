#ifndef SR_NW
#define SR_NW

using namespace std;

void save_csr(sparse_csr_weighted csr_info,string &file_name){
  ofstream file1;
  file1.open(file_name.c_str());
  file1<<*csr_info.number_of_nodes<<endl;
  for(int i=0;i<*csr_info.number_of_nodes;i++){
    file1<<csr_info.confidence[i]<<" ";
  }
  file1<<endl;
  for(int i=0;i<*csr_info.number_of_nodes+1;i++){
    file1<<csr_info.row_ptr[i]<<" ";
  }
  file1<<endl;
  for(int i=0;i<csr_info.row_ptr[*csr_info.number_of_nodes];i++){
    file1<<csr_info.influence[i]<<" ";
  }
  file1<<endl;
  for(int i=0;i<csr_info.row_ptr[*csr_info.number_of_nodes];i++){
    file1<<csr_info.col_index[i]<<" ";
  }
  file1<<endl;
  file1.close();
}

void save_info(network_info info1, string &file_name, n_nodes n){
  ofstream file3;
  file3.open(file_name.c_str());
  for(int i=0;i<n;i++){
    file3<<info1.nodes_types[i]<<" ";
  }
  file3<<endl;
  file3<<(int)*info1.time_length<<endl;
  file3<<*info1.p_threshold<<endl;
  file3<<*info1.n_threshold<<endl;
  file3.close();
}

sparse_csr_weighted read_csr(string &file_name){
  ifstream file2;
  file2.open(file_name.c_str());
  sparse_csr_weighted csr_info2;
  csr_info2.number_of_nodes=new n_nodes();
  file2>>*csr_info2.number_of_nodes;
  csr_info2.confidence=new double[*csr_info2.number_of_nodes];
  for(int i=0;i<*csr_info2.number_of_nodes;i++){
    file2>>csr_info2.confidence[i];
  }
  csr_info2.row_ptr=new int[*csr_info2.number_of_nodes+1];
  for(int i=0;i<*csr_info2.number_of_nodes+1;i++){
    file2>>csr_info2.row_ptr[i];
  }
  csr_info2.influence= new double[csr_info2.row_ptr[*csr_info2.number_of_nodes]];
  for(int i=0;i<csr_info2.row_ptr[*csr_info2.number_of_nodes];i++){
    file2>>csr_info2.influence[i];
  }
  csr_info2.col_index=new int[csr_info2.row_ptr[*csr_info2.number_of_nodes]];
  for(int i=0;i<csr_info2.row_ptr[*csr_info2.number_of_nodes];i++){
    file2>>csr_info2.col_index[i];
  }
  file2.close();
  return csr_info2;
}

network_info read_info(string &file_name, n_nodes n){
  ifstream file4;
  file4.open(file_name.c_str());
  network_info info4;
  info4.nodes_types=new node_type[n];
  int temp;
  for (int i=0;i<n;i++){
      file4>>temp;
      info4.nodes_types[i] = (node_type)temp;
  }
  info4.time_length=new uint8_t();
  file4>>temp;
  *info4.time_length = temp;
  info4.n_threshold=new double();
  info4.p_threshold=new double();
  file4>>*info4.p_threshold;
  file4>>*info4.n_threshold;
  file4.close();
  return info4;
}

void save_network(network_in_device &nw_device, string &info_file_name, string &csr_file_name){
  save_csr(nw_device.csr_info, csr_file_name);
  save_info(nw_device.nw_info, info_file_name, *nw_device.csr_info.number_of_nodes);
  return;
}

network_in_device read_network(string &info_file_name, string &csr_file_name){
  network_in_device nw_device;
  nw_device.csr_info = read_csr(csr_file_name);
  nw_device.nw_info = read_info(info_file_name, *nw_device.csr_info.number_of_nodes);
  return nw_device;
}

#endif