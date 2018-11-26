% Assume the first column is user_id,
% the second and the third columns are about candidates_id,
% the last column is outcome y
clear
clc


load iqadata_ref1.txt;
Pair_Compar=iqadata_ref1;
data = Pair_Compar(Pair_Compar(:,4)~=0,:);

kappa =20;
nt = 50;
trate = 50;
model = 1; %% 1 for Bradley and 2 for Thurstone
for run_time=1:20
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
result = lbi_likelihood(d(train,:),x(train,:),y(train),kappa,[],[],nt,trate,model,group,1);
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
    fit = lbi_likelihood(d(train(folds~=i),:),x(train(folds~=i),:),y(train(folds~=i)),kappa,alpha,t,[],[],model,group);
    i
    toc()
    res = (y(train(folds==i))*ones(1,nt)).*(d(train(folds==i),:)*fit.s_path + x(train(folds==i),:)*fit.path);
    residmat(i,:) = (1-mean(sign(res)))/2;
end
cv_error = mean(residmat);
cv_sd = sqrt(var(residmat)/K);
errorbar(1:nt,cv_error,cv_sd,cv_sd);

%%% Determine the optimal choice
k = find(cv_error==min(cv_error));
k=k(1);
res = y(test).*(d(test,:)*result.s_path(:,k(1))+x(test,:)*result.path(:,k(1)));
test_error(1,run_time)= (1-mean(sign(res)))/2;
end


a=min(test_error);
b=mean(test_error);
c=max(test_error);
d=std(test_error);
 
