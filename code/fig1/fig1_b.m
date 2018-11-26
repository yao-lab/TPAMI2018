
close all;
clear;
clc;
load movie_data.mat;
data = Train_User_Comparsion;
F=part_movie_genre(:,2:end);
N_F = size(F,2); % number of feature
model_solve=2;
kappa = 20;
nt = 50;
trate = 100;

[ind,w]=sort(data(:,4),'ascend');
data=data(w,:);
m = size(data,1); %% number of comparison
n = max(max(data(:,5:6)));  %% number of item
p = max(data(:,4)); %% number of user



u = data(:,4);
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
%x_g = sparse(1:m,u,ones(1,m),m,p);
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

errorbar(1:nt,cv_error,cv_sd,cv_sd);
%%% Determine the optimal choice
k = find(cv_error==min(cv_error));
k=k(1);
 
beta=result.path(1:N_F,k);
delta=result.path(N_F+1:end,k);
temp=reshape(delta,N_F,p);
common_group=beta'*F';
s_group=zeros(length(unique(data(:,4))),n);
[ind w0]=sort(common_group,'descend');
for i=1:length(unique(data(:,4)))
    s_group(i,:)=beta'*F'+temp(:,i)'*F';
end
kendall_distance=zeros(length(unique(data(:,4))),1);
for j=1:1:length(unique(data(:,4)))
    
    kendall_distance(j,1)=corr(common_group',s_group(j,:)','type','Kendall');
end
[ind w]=sort(kendall_distance,'descend');

delta_sum=zeros(p+1,50);
for i=1:50
    delta_1=result.path(N_F+1:end,i);
    beta=result.path(1:N_F,i);
    temp=reshape(delta_1,N_F,p);
    for j=1:p+1
        if j<=p
            delta_sum(j,i)=sum(abs(temp(:,j)));  %L1 norm
        else
            delta_sum(j,i)=sum(abs(beta));
        end
    end
end

t=result.tlist;
a =zeros(p+1,1)+Inf;
for i = length(t):-1:1
    a(delta_sum(:,i)~=0,1)= t(i)
end

[position_id position_index]=sort(a(:,1));
detected1=[position_index position_id];
user_id=detected1(:,1);

cc=[22];
dd=[8 18 2];
tt1=[3     1    15     5    17    11     7    21     6    12     4    14    13    10    19];
tt2=[16 20 9];
temptotal=[dd tt1 tt2 cc];
path=[zeros(p+1,1),delta_sum];
t = [0,result.tlist];
j=0;
for  i=1:22
    user_selected=temptotal(i);
    if ismember(user_selected,dd)
        type = 'r';
    else if ismember(user_selected,tt1)
            type = 'g';
        else if ismember(user_selected,tt2)
                type = 'b';
            else if ismember(user_selected,cc)
                    type = 'm';
                end
            end
        end
    end
    figure(1);
    plot(t,path(user_selected,:),type);
    hold on;
end

beta=result.path(1:N_F,k);
delta=result.path(N_F+1:end,k);
temp=reshape(delta,N_F,p);
common_group=beta'*F';
s_group=zeros(length(unique(data(:,4))),n);
[ind w0]=sort(common_group,'descend');
for i=1:length(unique(data(:,4)))
    s_group(i,:)=beta'*F'+temp(:,i)'*F';
end
