% Assume the first column is user_id,
% the second and the third columns are about candidates_id,
% the last column is outcome y
clear
clc
load univer_data.mat;
Pair_Compar=univer_qqxu;
data =  Pair_Compar;


kappa =20;
nt = 50;
trate = 50;
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
delta_1=result.path(1:n*p,k);

temp=reshape(delta_1,p,n);
preference_bias_user=[];
for i=1:p
    deltas_sum=sum(temp(i,:));
    if deltas_sum~=0
        preference_bias_user=[preference_bias_user
            i];
    end
end
gamma=result.path(n*p+1:end,k);
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

data=Pair_Compar;
detected=preference_detected;
jk=length(detected);
dd=preference_detected(1:jk);

temptotal=[dd];

path=[zeros(p,1),delta_sum];
t = [0,result.tlist];
j=0;
for  i=1:jk
    user_selected=temptotal(i);
    if ismember(user_selected,dd)
        type = 'r';
        
    end
    figure(1);
    plot(t,path(user_selected,:),type);
    str=num2str(user_selected);
    temp1=strcat('ID=',str);
    
    if any(dd==user_selected)
        j=j+1;
        %text(t(end-j),path(user_selected,end-j),temp1,'fontsize',14);
    end
    hold on;
end
xlabel('t','fontsize',20)
ylabel('\delta','fontsize',20); set(gca, 'FontSize', 16); hold on;
yl=ylim;
plot([t(k),t(k)],[yl(1), yl(2)],'r:');text(t(k)+0.02,yl(1),'t_{cv}','fontsize',12,'color','r');hold on;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%position bias

data=Pair_Compar;
detected=position_detected;

tt3=[detected(end-4,1),detected(end-3,1),detected(end-2,1),detected(end-1,1),detected(end,1)];

dd=position_detected(1:10);
aa=tt3;
temptotal=[dd aa];
path = [zeros(p,1),result.path(n*p+1:end,:)];

t = [0,result.tlist];
j=0;
for  i=1:15
    user_selected=temptotal(i);
    if ismember(user_selected,dd)
        type = 'r';
    else if ismember(user_selected,tt3)
            type = 'b';
            
        end
    end
    figure(2);
    plot(t,path(user_selected,:),type);
    str=num2str(user_selected);
    temp1=strcat('ID=',str);
    
    if any(dd==user_selected)
        j=j+1;
        text(t(end-j),path(user_selected,end-j),temp1,'fontsize',14);
    end
    hold on;
end
xlabel('t','fontsize',20);
ylabel('\gamma','fontsize',20);set(gca, 'FontSize', 16); hold on;
yl=ylim;
plot([t(k),t(k)],[yl(1), yl(2)],'r:');text(t(k)+0.02,yl(1),'t_{cv}','fontsize',12,'color','r');hold on;% ylabel('\delta','fontsize',24)


%%% training models ===model 1

load univer_data.mat;
Pair_Compar=univer_qqxu;
data =  Pair_Compar;
data = Pair_Compar(Pair_Compar(:,4)~=0,:);
test_error=zeros(1,20);

kappa =20;
nt = 50;
trate = 50;
model = 1; %% 1 for Bradley and 2 for Thurstone

m = size(data,1); %% number of comparison
n = max(max(data(:,2:3)));  %% number of item
p = max(data(:,1)); %% number of user

u = data(:,1);
i = data(:,2);
j = data(:,3);
y = data(:,4);

%%%   y = d * s + x1 * delta + x2 * gamma
d = sparse([1:m,1:m],[i;j],[ones(1,m),-ones(1,m)],m,n);
x1 = sparse([1:m,1:m],[u+(i-1)*p,u+(j-1)*p],[ones(1,m),-ones(1,m)],m,p*n);
x2 = sparse(1:m,u,ones(1,m),m,p);
x = [x1,x2];

%%% split the training and test
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

%% Cross-validation
K =5;
folds = mod(randperm(length(train)),K)+1;

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


%% Determine the optimal choice
k = find(cv_error==min(cv_error));
k=k(1);
delta_1=result.path(1:n*p,k);
temp=reshape(delta_1,p,n);
preference_bias_user=[];
for i=1:p
    deltas_sum=sum(temp(i,:));
    if deltas_sum~=0
        preference_bias_user=[preference_bias_user
            i];
    end
end
gamma=result.path(n*p+1:end,k);
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

data=Pair_Compar;
detected=preference_detected;


j=length(detected);

tt1=[detected(floor((j-3)/2),1),detected(floor((j-1)/2),1),detected(floor((j+1)/2),1)];
tt2=[detected(end-2,1),detected(end-1,1),detected(end,1)];

 
dd=preference_detected(1:9);

aa=[tt1 tt2];
temptotal=[dd aa];
 
path=[zeros(p,1),delta_sum];
t = [0,result.tlist];
j=0;
for  i=1:15
    user_selected=temptotal(i);
    if ismember(user_selected,dd)
        type = 'r';
    else if ismember(user_selected,tt1)
            type = 'g';
        else if ismember(user_selected,tt2)
                type = 'b';
            end
        end
    end
    figure(3);
    plot(t,path(user_selected,:),type);
    str=num2str(user_selected);
    temp1=strcat('ID=',str);
    
    if any(dd==user_selected)
        j=j+1;
        %text(t(end-j),path(user_selected,end-j),temp1,'fontsize',14);
    end
    hold on;
end
xlabel('t','fontsize',20)
ylabel('\delta','fontsize',20); set(gca, 'FontSize', 16); hold on;
yl=ylim;
plot([t(k),t(k)],[yl(1), yl(2)],'r:');text(t(k+1)+0.02,yl(1),'t_{cv}','fontsize',12,'color','r');hold on;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%position bias


 data=Pair_Compar;
detected=position_detected;

tt3=[detected(end-4,1),detected(end-3,1),detected(end-2,1),detected(end-1,1),detected(end,1)];
 
dd=position_detected(1:10);
aa=tt3;
temptotal=[dd aa];
path = [zeros(p,1),result.path(n*p+1:end,:)];
path=[zeros(p,1),delta_sum];
t = [0,result.tlist];
j=0;
for  i=1:15
    user_selected=temptotal(i);
    if ismember(user_selected,dd)
        type = 'r';
    else if ismember(user_selected,tt3)
            type = 'b';
            
        end
    end
    figure(4);
    plot(t,path(user_selected,:),type);
    str=num2str(user_selected);
    temp1=strcat('ID=',str);
    
    if any(dd==user_selected)
        j=j+1;
        %text(t(end-j),path(user_selected,end-j),temp1,'fontsize',14);
    end
    hold on;
end
xlabel('t','fontsize',20)
ylabel('\gamma','fontsize',20);set(gca, 'FontSize', 16); hold on;
yl=ylim;
plot([t(k),t(k)],[yl(1), yl(2)],'r:');text(t(k)+0.02,yl(1),'t_{cv}','fontsize',12,'color','r');hold on;

clear;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% training models ===model 2

load univer_data.mat;
Pair_Compar=univer_qqxu;
data =  Pair_Compar;
data = Pair_Compar(Pair_Compar(:,4)~=0,:); test_error=zeros(1,20);

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

%%   y = d * s + x1 * delta + x2 * gamma
d = sparse([1:m,1:m],[i;j],[ones(1,m),-ones(1,m)],m,n);
x1 = sparse([1:m,1:m],[u+(i-1)*p,u+(j-1)*p],[ones(1,m),-ones(1,m)],m,p*n);
x2 = sparse(1:m,u,ones(1,m),m,p);
x = [x1,x2];

%% split the training and test
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

% Cross-validation
K =5;
folds = mod(randperm(length(train)),K)+1;

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


% Determine the optimal choice
k = find(cv_error==min(cv_error));
k=k(1);
delta_1=result.path(1:n*p,k);
temp=reshape(delta_1,p,n);
ttt=temp;
preference_bias_user=[];
for i=1:p
    deltas_sum=sum(temp(i,:));
    if deltas_sum~=0
        preference_bias_user=[preference_bias_user
            i];
    end
end
gamma=result.path(n*p+1:end,k);
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

data=Pair_Compar;
detected=preference_detected;

commom_score=result.s_path(:,k);

j=length(detected);
jk=length(detected);
% tt1=[detected(floor((j-3)/2),1),detected(floor((j-1)/2),1),detected(floor((j+1)/2),1)];
% tt2=[detected(end-2,1),detected(end-1,1),detected(end,1)];

 
dd=preference_detected(1:9);

temptotal=[dd];
aa=setdiff(a,dd) ;
% path = [zeros(p,1),result.path(n*p+1:end,:)];
path=[zeros(p,1),delta_sum];
t = [0,result.tlist];
j=0;
for  i=1:9
    user_selected=temptotal(i);
    if ismember(user_selected,dd)
        type = 'r';
        
    end
    figure(5);
    plot(t,path(user_selected,:),type);
    str=num2str(user_selected);
    temp1=strcat('ID=',str);
    
    if any(dd==user_selected)
        j=j+1;
        %text(t(end-j),path(user_selected,end-j),temp1,'fontsize',14);
    end
    hold on;
end
xlabel('t','fontsize',20)
ylabel('\delta','fontsize',20); set(gca, 'FontSize', 16); hold on;
yl=ylim;
plot([t(k),t(k)],[yl(1), yl(2)],'r:');text(t(k)+0.02,yl(1),'t_{cv}','fontsize',12,'color','r');hold on;

data=Pair_Compar;
detected=position_detected;
tt3=[detected(end-4,1),detected(end-3,1),detected(end-2,1),detected(end-1,1),detected(end,1)];
dd=position_detected(1:10);
aa=tt3;
temptotal=[dd aa];
path = [zeros(p,1),result.path(n*p+1:end,:)];
 
t = [0,result.tlist];
j=0;
for  i=1:15
    user_selected=temptotal(i);
    if ismember(user_selected,dd)
        type = 'r';
    else if ismember(user_selected,tt3)
            type = 'b';
            
        end
    end
    figure(6);
    plot(t,path(user_selected,:),type);
    str=num2str(user_selected);
    temp1=strcat('ID=',str);
    
    if any(dd==user_selected)
        j=j+1;
        text(t(end-j),path(user_selected,end-j),temp1,'fontsize',14);
    end
    hold on;
end
xlabel('t','fontsize',20)
ylabel('\gamma','fontsize',20);set(gca, 'FontSize', 16); hold on;
yl=ylim;
plot([t(k),t(k)],[yl(1), yl(2)],'r:');text(t(k)+0.02,yl(1),'t_{cv}','fontsize',12,'color','r');hold on;
 
