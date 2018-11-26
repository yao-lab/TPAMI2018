function fit = lbi_likelihood(d,X,y,kappa,alpha,tlist,nt,trate,model,group,print)
if nargin < 4
    error('more input arguments needed.');
end
if nargin < 5 || isempty(alpha)
    sigma = svds([d,X],1);
    alpha = length(y)/kappa/sigma^2;
end
if nargin < 7 || isempty(nt)
    nt = 100;
end
if nargin < 8 || isempty(trate)
    trate = 0.01;
end
if nargin < 9 || isempty(model)
    model = 1;
end
if nargin < 11
    print = 0;
end

tic();
[n,p] = size(X);
n_item = size(d,2);
beta = zeros(p,1);
z = zeros(p,1);
s = zeros(n_item,1);

if nargin < 10 || isempty(group)
    group = 1:p;
end
group_index = unique(group);

% Skip the 0 part
ngroup = length(group_index);
temp = zeros(1,ngroup);
t0 = 0;
while(1)
    t0 = t0 + alpha;
    [g_s,g_beta] = grad_likelihood(d,X,y,s,beta,model);
    s = s - g_s*(kappa*alpha/n);
    z = z - g_beta*(alpha/n);
    for i=1:ngroup
        temp(i) = norm(z(group==group_index(i)));
    end
    if any(temp>=1)
        break
    end
end
beta = shrinkage(z,group,group_index)*kappa;

if nargin < 6 || isempty(tlist)
    tlist = t0*trate.^((0:(nt-1))/(nt-1));
end
nt = length(tlist);
path = zeros(p,nt);
s_path = zeros(n_item,nt);
k = double(sum(tlist<=t0));
if k>0
    s_path(:,1:k) = s*ones(1,k);
    path(:,k) = beta;
end
k = k+1;

maxiter = ceil((tlist(nt) - t0)/alpha);
for iter = 1:maxiter
    [g_s,g_beta] = grad_likelihood(d,X,y,s,beta,model);
    s = s - g_s*(kappa*alpha/n);
    z = z - g_beta*(alpha/n);
    beta = shrinkage(z,group,group_index)*kappa;
    dt = (iter*alpha+t0-tlist(k));
    while (k<=nt && dt>0)
        path(:,k) = shrinkage(z+g_beta*(dt/n),group,group_index)*kappa;
        s_path(:,k) = s + g_s*(kappa*dt/n);;
        k = k+1;
        if (print)
            disp(strcat('Process:',num2str(100*iter/maxiter),'%, time cost ',num2str(toc()),' seconds.'));
        end
        if(k>nt)
            break;
        end
        dt = (iter*alpha+t0-tlist(k));
    end
end
fit.path = path;
fit.s_path = s_path;
fit.tlist = tlist;
fit.alpha = alpha;
end

%------------------------------------------------------------------
% End function
%------------------------------------------------------------------

function X = shrinkage(z,group,group_index)
X = z;
for i=1:length(group_index)
    group_i = find(group == group_index(i));
    temp = X(group_i);
    norm_i = norm(temp);
    X(group_i) = z(group_i)*max(1-1/norm_i,0);
end
end
%------------------------------------------------------------------
% End function
%------------------------------------------------------------------
function [g_s,g_beta] = grad_likelihood(d,X,y,s,beta,model)
res = (d*s+X*beta).*y;
if model==1
    grad = -y./(1+exp(res));
elseif model==2
    grad = -y.*exp(-res.^2/2)./normcdf(res)/sqrt(2*pi);
else
    error('Method must be in {1,2}.');
end
g_s = d'*grad;
g_beta = X'*grad;
end
%------------------------------------------------------------------
% End function
%------------------------------------------------------------------