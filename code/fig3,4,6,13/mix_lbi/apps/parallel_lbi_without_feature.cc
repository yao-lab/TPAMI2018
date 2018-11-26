#include <iostream>
#include "../include/parameters.h"
#include "../include/util.h"
#include "../include/lock.h"
#include "../include/barrier.h"
#include <thread>
#include <math.h>
#include <../lib/Eigen/Sparse>
#include <../lib/Eigen/Dense>
using namespace std;
using namespace Eigen;
typedef void (*pf)(int index, Params* params);
void sync_cyclic_lbi(Range range,Range range_2,Params* params,Barrier& Computation_Barrier, Barrier& Cache_Update_Barrier,Barrier& Break_Barrier);

//void lbi_linear_sync(int index, Params* params);
//void lbi_linear_sync_cache(int rank, Params* params);
//void lbi_linear_async(int index, Params* params);
//void async_cyclic_worker_lbi(pf algorithm, Range range, Params* params);
std::vector<VectorXf> y;
std::vector<SparseMatrix<float>> d;
  // define auxilary variables

VectorXf eta;
VectorXf g_eta;
MatrixXf update_eta;
std::vector<VectorXf> xi;
std::vector<VectorXf> z;
std::vector<VectorXf> g;
std::vector<double> gam;
std::vector<double> z_g;
std::vector<double> g_g;
MatrixXf x_path;
bool t_initial=true, inter,group;
VectorXf t_list;
double t_min = 0;
SpinLock lock_t_min;
int total_num_threads = 1;

int main(int argc, char *argv[]) {
  /*VectorXf u_ref = VectorXf::Random(200);
  A = MatrixXf::Random(500,200);
  b = A * u_ref + VectorXf::Random(500)*0.1;
  writeEigen(A, "A.mat");
  writeEigen(b, "b.mat");*/

  Params params;
  double kappa = 0.;
  //string data_file_name;
  //string label_file_name;
  //parse_input_argv_mm(&params, argc, argv, data_file_name, label_file_name, kappa);

  char* input_file_d;
  if (argc<2){
    cout << "Error" << endl;
    exit(-1);
  }else{
    input_file_d = argv[1];
    if (argc<3){
      params.kappa = 10.0;
    }else{
      params.kappa = atof(argv[2]);
    }
    if (argc<4){
      params.step_size = 1.0;
    }else{
      params.step_size = atof(argv[3]);
    }
    if (argc<5){
      params.total_num_threads = 4;
    }else{
      params.total_num_threads = atoi(argv[4]);
    }
    if (argc<6){
      params.inter = false;
    }else{
      params.inter = (atoi(argv[5])!=0);
    }
    if (argc<7){
      params.group = false;
    }else{
      params.group = (atoi(argv[6])!=0);
    }
    if (argc<8){
      params.nt = 100;
    }else{
      params.nt = atoi(argv[7]);
    }
    if (argc<9){
      params.trate = 100.0;
    }else{
      params.trate = atof(argv[8]);
    }
  }
  MatrixXf data = loadEigen<MatrixXf>(input_file_d);
  int user_size = data.col(0).maxCoeff();
  int item_size = data.col(1).maxCoeff();
  int temp = data.col(2).maxCoeff();
  if (temp>item_size) item_size = temp;
  int feature_size = item_size;

  inter = params.get_inter();
  group = params.group;
  std::vector<int> T_size = ReadProblem(data,&d,&y,user_size,item_size);
  xi.resize(user_size);
  z.resize(user_size);
  gam.resize(user_size);
  z_g.resize(user_size);
  g_g.resize(user_size);
  g.resize(user_size);
  for (int i=0; i<user_size; i++){
    xi[i] = VectorXf::Zero(feature_size);
    z[i] = VectorXf::Zero(feature_size);
    g[i] = VectorXf::Zero(feature_size);
    gam[i] = 0;
    z_g[i] = 0;
    //y[i] = Map<const Eigen::Matrix<double, T_size[i], 1>>(y_value[i].data());
  }
  int problem_size = feature_size + feature_size*user_size + inter*user_size;
  eta = VectorXf::Zero(feature_size);
  g_eta = VectorXf::Zero(feature_size);
  x_path = MatrixXf::Zero(problem_size,params.nt);
  t_list = VectorXf::Zero(params.nt);

  params.problem_size = feature_size;
  params.sample_size = user_size;
  params.item_size = item_size;
  //params.step_size = 500000.0/sample_size;
  //params.kappa = 0.001;
  //params.max_itrs = 100;
  //params.total_num_threads = 2;
  // Step 0: Define the params and input file names  
  //double lambda = 0.;
  //string data_file_name;
  //string label_file_name;
  // Step 1. Parse the input argument
  //parse_input_argv_mm(&params, argc, argv, data_file_name, label_file_name, lambda);

  // Step 2. Load the data or generate synthetic data, define matained variables
  
  //loadMarket(A, data_file_name);
  //loadMarket(b, label_file_name);

  // set para
  total_num_threads = params.total_num_threads;
  std::vector<std::thread> mythreads;
  int num_workers = total_num_threads;
  double block_size = double(user_size)/double(num_workers);
  params.block_size = block_size;
  if (block_size<1){
    params.total_num_threads = user_size;
    total_num_threads = user_size;
    num_workers = user_size;
    block_size = 1;
    params.block_size = 1;
  }
  double block_size_2 = double(feature_size)/double(num_workers);
  /*if (block_size_2<1){
    params.total_num_threads = feature_size;
    total_num_threads = feature_size;
    num_workers = feature_size;
    block_size_2 = 1;
  }*/
  update_eta = MatrixXf::Zero(feature_size,num_workers);
  Barrier computation_barrier(num_workers);
  Barrier cache_update_barrier(num_workers);
  Barrier break_barrier(num_workers);
  double start_time = get_wall_time();
  for (size_t i = 0; i < num_workers; i++) {
      // partition the indexes
      Range range(int(ceil(i * block_size)), int(ceil((i + 1) * block_size)));
      if (i == num_workers - 1) {
        range.end = user_size;
      }
      Range range_2(int(ceil(i * block_size_2)), int(ceil((i + 1) * block_size_2)));
      if (i == num_workers - 1) {
        range_2.end = feature_size;
      }
      // cyclic coordinate update
      //mythreads.push_back(std::thread(async_cyclic_worker_lbi,&lbi_linear_async,range, &params));
      mythreads.push_back(std::thread(sync_cyclic_lbi,range,range_2,&params,std::ref(computation_barrier),std::ref(cache_update_barrier),std::ref(break_barrier)));
  }
  for (size_t i = 0; i < total_num_threads; i++) {
    mythreads[i].join();
  }
  double end_time = get_wall_time();
  // Step 7. Print results
  //print_params(params);
  //cout << "Objective value is: " << objective(Atx, b, x, lambda) << endl;
  //cout << "max_itrs: " << params.max_itrs << endl;
  //cout << "Used Objective value is: " << (b - Ax).squaredNorm()*0.5 << endl;
  //cout << "True Objective value is: " << (b - A * x).squaredNorm()*0.5 << endl;
  //cout << "||Atx- A*x||/||b||: " << (Ax - A * x).norm()/b.norm() << endl;
  cout << "Computing time  is: " << end_time - start_time << endl;
  cout << "---------------------------------" << endl;
  // cout << params.step_size << endl;
  // cout << params.total_num_threads << endl;
  // cout << "---------------------------------" << endl;
  writeEigen(x_path, "Output.txt");
  return 0;
}

void sync_cyclic_lbi(Range range,Range range_2, Params* params,
                     Barrier& Computation_Barrier, Barrier& Cache_Update_Barrier,Barrier& Break_Barrier) {
  int rank = floor(double(range.start) / double(params->block_size)); 
  //auto id = std::this_thread::get_id();
  int nt = params->nt;
  int feature_size = params->problem_size;
  int user_size = params->sample_size;
  double step_size = params->get_step_size();
  double kappa = params->get_kappa();
  double t = 0, t_1 = 0,t_2 = 0,trate,rate;
  int k = 0;
  double start_time = get_wall_time();
  while (true){
    double start_time = get_wall_time();
    t += step_size;
    for (int i = range.start; i < range.end; i++){
      VectorXf res = d[i]*(eta + xi[i]) - y[i];
      if (inter){
        res.array() += gam[i];
        g_g[i] = res.sum();
        z_g[i] -= step_size*g_g[i];
        gam[i] = kappa*shrinkage(z_g[i],t_initial,&t_2,g_g[i]);
      }
      g[i] = d[i].transpose()*res;
      update_eta.col(rank) += g[i];
      z[i] -= step_size*g[i];
      xi[i] = kappa*shrinkage(z[i],t_initial,group,&t_1,g[i]);
      if (inter){
        t_1 = t_1>t_2?t_2:t_1;
        t_2 = 0;
      }
      if (t_initial & t_1<0){
        double t_temp = t +  t_1;
        {
          std::lock_guard<SpinLock> lock(lock_t_min);
          if(t_min ==0 | t_temp < t_min){
            t_min = t_temp;
          }
        }
      }
      t_1 = 0;
    }
    Computation_Barrier.wait();
    int col_feature = range_2.end-range_2.start;
    g_eta.segment(range_2.start,col_feature) = update_eta.middleRows(range_2.start,col_feature).rowwise().sum();
    eta.segment(range_2.start,col_feature) -= step_size*g_eta.segment(range_2.start,col_feature);
    update_eta.middleRows(range_2.start,col_feature) = MatrixXf::Zero(col_feature,total_num_threads);
    if (rank==0){
      if(t_initial & t_min > 0){
        t_initial = false;
        trate = params->trate;
        rate = pow(trate,1.0/(nt-1));
        for (int i = 0; i < nt; i++){
          t_list(i) = t_min;
          t_min *= rate;
        }
      }
    }
    Cache_Update_Barrier.wait();
    if(!t_initial){
      while(t>t_list(k)){
        double delta_t = t - t_list(k);
        x_path.block(range_2.start, k, col_feature, 1) = eta.segment(range_2.start,col_feature) + g_eta.segment(range_2.start,col_feature)*delta_t;
        for (int i = range.start; i < range.end; i++){
          x_path.block((i+1)*feature_size, k, feature_size, 1) = kappa*shrinkage(z[i]+g[i]*delta_t,t_initial,group,&t_1,g[i]);
          if (inter){
            x_path((user_size+1)*feature_size+i, k) = kappa*shrinkage(z_g[i]+g_g[i]*delta_t,t_initial,&t_2,g_g[i]);
          }
        }
        k++;
        //cout << "Computing time  is: " << end_time - start_time << endl;
        //cout << "---------------------------------" << endl;
        if (k>=nt) break;
      }
    }
    if (k>=nt) break;
    Break_Barrier.wait();
  }
}
    


