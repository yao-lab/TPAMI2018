close all;
clear;
clc;
load movie_data.mat; 

time_number=20;
lb_error=zeros(3,time_number);
hodge_error=zeros(1,time_number);
for times=1:time_number
    
    data = Train_User_Comparsion;
    kappa =20;
    nt = 50;
    trate =100;
    
    m = size(data,1); %% number of comparison
    n = max(max(data(:,5:6)));  %% number of item
    n_item=n;
    p = max(data(:,1)); %% number of user
    u = data(:,1);
    ii = data(:,5);
    jj = data(:,6);
    temp=randperm(n_item); 
    for i=1:length(ii)
        ii(i)=temp(ii(i));
        jj(i)=temp(jj(i));
    end
    i = ii;
    j =jj;
    y = data(:,7);
    
    [ind w]=sort(temp,'ascend');
    F=part_movie_genre(w,2:end);
    N_F = size(F,2);
 
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
    K = 4;
    p_test = 1/K;
    p_train = 1-p_test;
    test = (ii>n_item*p_train|jj>n_item*p_train);
    train = find(~test);
    n_train = floor(n_item*p_train);
    
    %%%% training models
    % LB
    tic()
    result_linear = lb_xqq_2(x1(train,:),x2(train,:),y(train),kappa,[],[],nt,trate,1,1);
    result_BT = lb_xqq_2(x1(train,:),x2(train,:),y(train),kappa,[],[],nt,trate,1,2);
    result_TM = lb_xqq_2(x1(train,:),x2(train,:),y(train),kappa,[],[],nt,trate,1,3);
    toc()
    t_linear = result_linear.tlist;
    alpha_linear = result_linear.alpha;
    
    t_BT = result_BT.tlist;
    alpha_BT = result_BT.alpha;
    
    t_TM = result_TM.tlist;
    alpha_TM = result_TM.alpha;
    %%% Cross-validation
    
    % LB
    residmat_linear = zeros(K,nt);
    residmat_BT = zeros(K,nt);
    residmat_TM= zeros(K,nt);
    
    for k=1:K
        k
        tic()
        aa = ii(train)/n_train;
        bb = jj(train)/n_train;
        fold_k = (aa<k*p_test & aa>((k-1)*p_test)) | (bb<k*p_test & bb>((k-1)*p_test));
        fit_linear = lb_xqq_2(x1(train(~fold_k),:),x2(train(~fold_k),:),y(train(~fold_k)),kappa,alpha_linear,t_linear,[],[],1,1);
        fit_BT = lb_xqq_2(x1(train(~fold_k),:),x2(train(~fold_k),:),y(train(~fold_k)),kappa,alpha_BT,t_BT,[],[],1,2);
        fit_TM = lb_xqq_2(x1(train(~fold_k),:),x2(train(~fold_k),:),y(train(~fold_k)),kappa,alpha_TM,t_TM,[],[],1,3);
        
        toc()
        res_linear = y(train(fold_k))*ones(1,nt).*([x1(train(fold_k),:),x2(train(fold_k),:)]*fit_linear.path);
        res_BT = y(train(fold_k))*ones(1,nt).*([x1(train(fold_k),:),x2(train(fold_k),:)]*fit_BT.path);
        res_TM = y(train(fold_k))*ones(1,nt).*([x1(train(fold_k),:),x2(train(fold_k),:)]*fit_TM.path);
        residmat_linear(k,:) = (1-mean(sign(res_linear)))/2;
        residmat_BT(k,:) = (1-mean(sign(res_BT)))/2;
        residmat_TM(k,:) = (1-mean(sign(res_TM)))/2;
    end
    cv_error_linear = mean(residmat_linear);
    cv_sd_linear = sqrt(var(residmat_linear)/K);
    
    cv_error_BT = mean(residmat_BT);
    cv_sd_BT = sqrt(var(residmat_BT)/K);
    
    cv_error_TM = mean(residmat_TM);
    cv_sd_TM = sqrt(var(residmat_TM)/K);
    
    errorbar(1:nt,cv_error_linear,cv_sd_linear,cv_sd_linear);
    %%% Determine the optimal choice
    k_linear = find(cv_error_linear==min(cv_error_linear));
    k_linear=k_linear(1);
    
    k_BT = find(cv_error_BT==min(cv_error_BT));
    k_BT=k_BT(1);
    
    k_TM = find(cv_error_TM==min(cv_error_TM));
    k_TM=k_TM(1);
    
    res_linear = y(test).* ([x1(test,:),x2(test,:)]*result_linear.path(:,k_linear(1)));
    test_error_linear = (1-mean(sign(res_linear)))/2;
    lb_error(1,times)=test_error_linear
    
    res_BT = y(test).* ([x1(test,:),x2(test,:)]*result_BT.path(:,k_BT(1)));
    test_error_BT = (1-mean(sign(res_BT)))/2;
    lb_error(2,times)=test_error_BT
    
    res_TM = y(test).* ([x1(test,:),x2(test,:)]*result_TM.path(:,k_TM(1)));
    test_error_TM = (1-mean(sign(res_TM)))/2;
    lb_error(3,times)=test_error_TM
    
    
    %%%%%%%%%%%%%%%%%%%%%Hodge error
    s = pinv(x1(train,:)'*x1(train,:))*(x1(train,:)'*y(train,:));
    res = y(test).*(x1(test,:)*s);
    hodge_test_error =(1-mean(sign(res)))/2;
    hodge_error(1,times)=hodge_test_error
end


hodge_min=min(hodge_error(1,:));
hodge_mean=mean(hodge_error(1,:));
hodge_max=max(hodge_error(1,:));
hodge_std=std(hodge_error(1,:));


linear_min=min(lb_error(1,:));
linear_mean=mean(lb_error(1,:));
linear_max=max(lb_error(1,:));
linear_std=std(lb_error(1,:));

bt_min=min(lb_error(2,:));
bt_mean=mean(lb_error(2,:));
bt_max=max(lb_error(2,:));
bt_std=std(lb_error(2,:));

tm_min=min(lb_error(3,:));
tm_mean=mean(lb_error(3,:));
tm_max=max(lb_error(3,:));
tm_std=std(lb_error(3,:));

result_leave_item_out=[ hodge_min  hodge_mean hodge_max hodge_std;
    linear_min linear_mean linear_max linear_std;
    bt_min bt_mean bt_max bt_std;
    tm_min tm_mean tm_max tm_std];
