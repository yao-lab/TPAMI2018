function fit = lb_xqq(d,X,y,kappa,alpha,tlist,nt,trate,group,print)

if nargin < 4
    error('more input arguments needed.');
end

if nargin < 5 || isempty(alpha)
    sigma = svds(X,1);
    alpha = length(y)/kappa/sigma^2;
end
if nargin < 10
    print = 0;
end

tic();
[U,S,V] = svd(full(d'*d));
s = diag(S);
s(s<=1e-8) = 0;
s(s>1e-8) = 1./s(s>1e-8);
inv_dd = U*diag(s)*V';

[n,p] = size(X);
n_item = size(d,2);
beta = zeros(p,1);
z = zeros(p,1);
path = zeros(p,nt);
s_path = zeros(n_item,nt);
group_index = unique(group);

% Skip the 0 part
res = X*beta - y;
s = -inv_dd*(d'*res);
res = res + d*s;
g = X'*res;
ngroup = length(group_index);
temp = zeros(1,ngroup);
for i=1:ngroup
    temp(i) = norm(g(group==group_index(i)));
end 
t0 = n/max(temp);
z = z - g*(t0/n);

if nargin < 6 || isempty(tlist)
    tlist = t0*trate.^((0:(nt-1))/(nt-1));
end
nt = length(tlist);
k = sum(tlist<=t0)+1;
s_path(:,1:(k-1)) = s*ones(1,k-1);

maxiter = ceil((tlist(nt) - t0)/alpha);
for iter = 1:maxiter
    res = X*beta - y;
    s = -inv_dd*(d'*res);
    res = res + d*s;
    g = X'*res;
    z = z - g*(alpha/n);
    beta = shrinkage(z,group,group_index)*kappa;
    dt = (iter*alpha+t0-tlist(k));
    while (k<=nt && dt>0)
        path(:,k) = shrinkage(z+g*(dt/n),group,group_index)*kappa;
        s_path(:,k) = s;
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