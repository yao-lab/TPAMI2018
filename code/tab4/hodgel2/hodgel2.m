% Assume the first column is user_id, 
% the second and the third columns are about candidates_id, 
% the last column is outcome y 
clear
clc

load iqadata_ref1.txt;
Pair_Compar=iqadata_ref1;
data = Pair_Compar;
   
time_number=20;
kappa =20;
nt = 50;
trate = 50;

lb_error=zeros(1,time_number);
hodge_error=zeros(1,time_number);
for times=1:time_number
m = size(data,1); %% number of comparison
n = max(max(data(:,2:3)));  %% number of item
p = max(data(:,1)); %% number of user

u = data(:,1);
i = data(:,2);
j = data(:,3);
y = data(:,4);

%%%%   y = d * s + x1 * delta + x2 * gamma
d = sparse([1:m,1:m],[i;j],[ones(1,m),-ones(1,m)],m,n);
x1 = sparse([1:m,1:m],[u+(i-1)*p,u+(j-1)*p],[ones(1,m),-ones(1,m)],m,p*n);
x2 = sparse(1:m,u,ones(1,m),m,p);
x = [x1,x2];

%%%% split the training and test
p_train = 0.7;
train = (rand(1,m)<p_train);
test = ~train;
train = find(train);

%%%% training models
% LB
group = (1:p)'*ones(1,n);
group = [group(:)',(p+1):(2*p)];
tic()
result = lb_xqq(d(train,:),x(train,:),y(train),kappa,[],[],nt,trate,group,1);
toc()
t = result.tlist;
alpha = result.alpha;


%%% Cross-validation
K =5;
folds = mod(randperm(length(train)),K)+1;
% LB
residmat = zeros(K,nt);
for i=1:K
    tic()
	fit = lb_xqq(d(train(folds~=i),:),x(train(folds~=i),:),y(train(folds~=i)),kappa,alpha,t,[],[],group);
	i
    toc()
    res = y(train(folds==i))*ones(1,nt).*( d(train(folds==i),:)*fit.s_path +x(train(folds==i),:)*fit.path);
	
  residmat(i,:) = (1-mean(sign(res)))/2;
end
cv_error = mean(residmat);
cv_sd = sqrt(var(residmat)/K);

errorbar(1:nt,cv_error,cv_sd,cv_sd);

%%% Determine the optimal choice
k = find(cv_error==min(cv_error));
k=k(1);

%%% Compute test error
res = y(test).* (d(test,:)*result.s_path(:,k(1)) + x(test,:)*result.path(:,k(1)));
test_error = (1-mean(sign(res)))/2;
lb_error(1,times)=test_error;

res = y(test).*(d(test,:)*result.s_path(:,1));
hodge_test_error =(1-mean(sign(res)))/2; 
hodge_error(1,times)=hodge_test_error;
end

a=min(hodge_error);
b=mean(hodge_error);
c=max(hodge_error);
d=std(hodge_error);

e=min(lb_error);
f=mean(lb_error);
g=max(lb_error);
h=std(lb_error);

final_results=[a b c d;
               e f g h];
 