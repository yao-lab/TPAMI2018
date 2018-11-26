
close all;
clear;
clc;
load movie_data.mat;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = Train_User_Comparsion;
F=part_movie_genre(:,2:end);
N_F = size(F,2);
kappa = 20;
nt = 50;
trate = 100;

[ind,w]=sort(data(:,3),'ascend');
data=data(w,:);
m = size(data,1); %% number of comparison
n = max(max(data(:,5:6)));  %% number of item
p = max(data(:,3)); %% number of user

model_solve=2;

u = data(:,3);
i = data(:,5);
j = data(:,6);
y = data(:,7);

d = sparse([1:m,1:m],[i;j],[ones(1,m),-ones(1,m)],m,n);
x1 = d * F;
x_d = [zeros(m,N_F*p)];
for i=1:p
    index = (u==i);
    x_d(index,(i-1)*N_F+(1:N_F)) = x1(index,:);
end

x_g=[];
x_d = sparse(x_d);
x2 = [x_d,x_g];

%%%% split the training and test

p_train = 1;
train = (rand(1,m)<p_train);
test = ~train;
train = find(train);

%%%% training models
% LB
tic()
result = lb_xqq_2(x1(train,:),x2(train,:),y(train),kappa,[],[],nt,trate,1,model_solve);
toc()
t = result.tlist;
alpha = result.alpha;


K =4;
fold_k = mod(randperm(length(train)),K)+1;
% LB
residmat = zeros(K,nt);
for i=1:K
    tic()
    fit = lb_xqq_2(x1(train(fold_k~=i),:),x2(train(fold_k~=i),:),y(train(fold_k~=i)),kappa,alpha,t,[],[],1,model_solve);
    i
    toc()
    res = y(train(fold_k==i))*ones(1,nt).*([x1(train(fold_k==i),:),x2(train(fold_k==i),:)]*fit.path);
    residmat(i,:) = (1-mean(sign(res)))/2;
end
cv_error = mean(residmat);
cv_sd = sqrt(var(residmat)/K);


k = find(cv_error==min(cv_error));
k=k(1);


beta=result.path(1:N_F,k);
delta=result.path(N_F+1:end,k);
temp=reshape(delta,N_F,p);
common_group=beta'*F';
s_group=zeros(length(unique(data(:,3))),n);
[ind w0]=sort(common_group,'descend');
for i=1:length(unique(data(:,3)))
    s_group(i,:)=beta'*F'+temp(:,i)'*F';
end
kendall_distance=zeros(length(unique(data(:,3))),1);
for j=1:1:length(unique(data(:,3)))
    
    kendall_distance(j,1)=corr(common_group',s_group(j,:)','type','Kendall');
end
[ind w]=sort(kendall_distance,'descend');

cc=[];
for i=1:7
    j=1:7;
    [ind w1]=sort(s_group(j(i),:),'descend');
    ind=w1(1:20);
    f=F(ind,:);
    his=sum(f,1);
    totalhis=sum(his);
    [a b]=sort(his./totalhis,'descend');
    cc=[cc
        a(1) b(1)
        a(2) b(2)];    
end

final_result=cc;



