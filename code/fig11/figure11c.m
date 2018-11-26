

%%%% training models ===model 1
% LB
clear;
clc;

load univer_data.mat;
Pair_Compar=univer_qqxu;
data = Pair_Compar(Pair_Compar(:,4)~=0,:);
 
kappa =20;
nt = 50;
trate = 50;
model = 2; %% 1 for Bradley and 2 for Thurstone

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
p_train = 1;
train = (rand(1,m)<p_train);
test = ~train;
train = find(train);

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
delta_1=result.path(1:n*p,k(1));

temp=reshape(delta_1,p,n);
t_bias=temp;
preference_bias_user=[];
for i=1:p
    deltas_sum=sum(temp(i,:));
    if deltas_sum~=0
        preference_bias_user=[preference_bias_user
            i];
    end
end
gamma=result.path(n*p+1:end,k(1));
temp2=find(gamma~=0);
position_bias_user=temp2;

delta_sum=zeros(p,50);
for i=1:50
    delta_1=result.path(1:n*p,i);
    temp=reshape(delta_1,p,n);
    for j=1:p
        delta_sum(j,i)=sum(abs(temp(j,:)));
    end
end
gamma2=result.path(n*p+1:end,:);

t=result.tlist;
a =zeros(p,2)+Inf;
for i = length(t):-1:1
    a(gamma2(:,i)~=0,1) = t(i)
    a(delta_sum(:,i)~=0,2)= t(i)
end

[position_id position_index]=sort(a(:,1));
detected1=[position_index position_id];
[preference_id preference_index]=sort(a(:,2));
detected2=[preference_index preference_id];
position_detected=detected1(1:length(position_bias_user),:);
preference_detected=detected2(1:length(preference_bias_user),:);

 
detected=preference_detected;
commom_score=result.s_path(:,k);
j=length(detected);
delta_top10(:,1)=commom_score+t_bias(detected(1,1),:)';
delta_top10(:,2)=commom_score+t_bias(detected(2,1),:)';
delta_top10(:,3)=commom_score+t_bias(detected(3,1),:)';
delta_top10(:,4)=commom_score+t_bias(detected(floor((j-3)/2),1),:)';
delta_top10(:,5)=commom_score+t_bias(detected(floor((j-1)/2),1),:)';
delta_top10(:,6)=commom_score+t_bias(detected(floor((j+1)/2),1),:)';
delta_top10(:,7)=commom_score+t_bias(detected(end-2,1),:)';
delta_top10(:,8)=commom_score+t_bias(detected(end-1,1),:)';
delta_top10(:,9)=commom_score+t_bias(detected(end,1),:)';
 
[aa, bb]=sort(commom_score,'descend');
ccc=1:1:261;
L2_ranking=[bb ccc' aa];
personal_top10=[];
for jj=1:9
    [aaa bbb]=sort(delta_top10(:,jj),'descend');
    personalized_ranking=[bbb aaa];
    temp3=[bbb ccc' aaa];
    temp2=[];
    for i=1:261
        for j=1:261
            if temp3(j,1)==bb(i)
                temp2=[temp2
                    temp3(j,2) temp3(j,3)];
            end
        end
    end
    personal_top10=[personal_top10 temp2];
end
ans=[L2_ranking personal_top10];
n=261;
item=1:1:n;
x=zeros(1,n)+1;
z=ans(:,2);
scatter(x,item,50,z,'filled');
hold on;
x=zeros(1,n)+2;
z=ans(:,4);
scatter(x,item,50,z,'filled');hold on;
x=zeros(1,n)+3;
z=ans(:,6);
scatter(x,item,50,z,'filled');hold on;
x=zeros(1,n)+4;
z=ans(:,8);
scatter(x,item,50,z,'filled');hold on;
 x=zeros(1,n)+5;
 z=ans(:,10);
  scatter(x,item,50,z,'filled');hold on;
x=zeros(1,n)+6;
z=ans(:,12);
scatter(x,item,50,z,'filled');hold on;
x=zeros(1,n)+7;
z=ans(:,14);
scatter(x,item,50,z,'filled');hold on;
x=zeros(1,n)+8;
z=ans(:,16);
scatter(x,item,50,z,'filled');hold on;
x=zeros(1,n)+9;
z=ans(:,18);
scatter(x,item,50,z,'filled');hold on;
 x=zeros(1,n)+10;
 z=ans(:,20);
 scatter(x,item,50,z,'filled');hold on;
