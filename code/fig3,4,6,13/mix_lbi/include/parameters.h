using namespace std;
struct Params {

  double step_size;
  double kappa;
  int max_itrs;
  int nt;
  double trate;
  double block_size;
  bool inter;
  bool sparse_feature;
  int total_num_threads;
  int item_size;
  int problem_size;
  int sample_size;
  bool group;
  double get_step_size() {
    return step_size;
  }

  double get_kappa() {
    return kappa;
  }

  bool get_inter() {
    return inter;
  }

  int get_problem_dimension() {
    return problem_size;
  }
  
  int get_sample_size() {
    return sample_size;
  }

 Params() : problem_size(0), max_itrs(100){}
  
};
