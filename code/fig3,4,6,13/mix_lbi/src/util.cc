#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <string>
#include <vector>
#include "util.h"

//  Windows
#ifdef _WIN32
#include <Windows.h>
double get_wall_time(){
  LARGE_INTEGER time,freq;
  if (!QueryPerformanceFrequency(&freq)){
    //  Handle error
    return 0;
  }
  if (!QueryPerformanceCounter(&time)){
    //  Handle error
    return 0;
  }
  return (double)time.QuadPart / freq.QuadPart;
}
double get_cpu_time(){
  FILETIME a,b,c,d;
  if (GetProcessTimes(GetCurrentProcess(),&a,&b,&c,&d) != 0){
    //  Returns total user time.
    //  Can be tweaked to include kernel times as well.
    return
        (double)(d.dwLowDateTime |
                 ((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
  }else{
    //  Handle error
    return 0;
  }
}
//  Posix/Linux
#else
#include <time.h>
#include <sys/time.h>
double get_wall_time(){
  struct timeval time;
  if (gettimeofday(&time,NULL)){
    //  Handle error
    return 0;
  }
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
double get_cpu_time(){
  return (double)clock() / CLOCKS_PER_SEC;
}
#endif

double shrinkage(double z, bool t_initial, double* t, double g){
  double x;
  t[0] = 0;
  if (z > 1){
    x = z-1;
  }else if (z < -1){
    x = z+1;
  }else{
    x = 0;
  }
  if (t_initial){
    t[0] = x/g;
  }
  return x;
}

VectorXf shrinkage(VectorXf z, bool t_initial, bool group, double* t, VectorXf g){
  VectorXf x = z;
  double t_temp = 0;
  t[0] = 0;
  if (!group){
    for (int i = 0; i < z.size(); i++){
      if (z(i) > 1){
        x(i) = z(i)-1;
      }else if (z(i) < -1){
        x(i) = z(i)+1;
      }else{
        x(i) = 0;
      }
      if (t_initial){
        t_temp = x(i)/g(i);
      }
      if (t_temp<t[0])
        t[0] = t_temp;
    }
  }else{
    double norm = z.norm();
    if (norm > 1){
      x = z*(1.0-1.0/norm);
    }else{
      x = z*0;
    }
    if (t_initial & norm > 1){
      double g_norm = g.dot(g);
      double g_z = g.dot(z);
      t[0] = (g_z+sqrt(g_z*g_z-g_norm*(norm*norm-1)))/g_norm;
    }
  }
  return x;
}

std::vector<int> ReadProblem(MatrixXf data,std::vector<SparseMatrix<float>>* d, std::vector<VectorXf>* y, int user_size, int item_size){
  int sample_size = data.rows();
  int average_sample = sample_size/user_size;
  int u = 0;
  typedef Eigen::Triplet<float> Triplet;
  std::vector<std::vector<Triplet>> T;
  std::vector<std::vector<float>> y_value;
  std::vector<int> T_size;
  T.resize(user_size);
  y_value.resize(user_size);
  T_size.resize(user_size);
  for (int i=0; i<user_size; i++){
    T[i].reserve(2*average_sample);
    y_value[i].reserve(average_sample);
    T_size[i] = 0;
  }
  for (int k=0; k<sample_size; k++){
    u = data(k,0)-1;
    int m = T_size[u];
    T_size[u]++;
    T[u].push_back(Triplet(m,data(k,1)-1,1));
    T[u].push_back(Triplet(m,data(k,2)-1,-1));
    y_value[u].push_back(float(data(k,3)));
  }
  for (int i=0; i<user_size; i++){
    SparseMatrix<float> A(T_size[i],item_size);
    A.setFromTriplets(T[i].begin(), T[i].end());
    (*d).push_back(A);
    //y[i] = Map<VectorXf>(y_value[i].data(),T_size[i]);
    VectorXf b = VectorXf::Zero(T_size[i]);
    for (int j=0; j<T_size[i]; j++)
      b(j) = y_value[i][j];
    (*y).push_back(b);
  }
  return T_size;
}
