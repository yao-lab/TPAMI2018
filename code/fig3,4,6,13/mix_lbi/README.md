This is the codes to reproduce the figures in our paper.

##### install ######
The installation needs c++11


##### Usage ######

## If feature matrix Phi is used
./bin/parallel_lbi_with_feature data_matrix Phi_matrix kappa alpha n_thread inter group sparse_feature nt trate

## Otherwise
./bin/parallel_lbi_without_feature data_matrix kappa alpha n_thread inter group nt trate

## Parameters
data_matrix: A four column matrix with each row (u,i,j,y), u is user index, (i,j) is a pair of item index, y is the comparison
Phi_matex: a n_item \times n_feature matrix
inter: whether position bias \gamma in included
group: whether use group penalty on beta
nt: number of t in t_list
trate: t_max/t_min