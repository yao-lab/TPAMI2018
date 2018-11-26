function fit = lb_xqq_2(X1,X2,y,kappa,alpha,tlist,nt,trate,print,model)

if nargin < 4
    error('more input arguments needed.');
end
X = [X1,X2];
if nargin < 5 || isempty(alpha)
    sigma = svds(X,1);
    alpha = length(y)/kappa/sigma^2;
end
if nargin < 9 || isempty(print)
    print = 0;
end

if nargin < 10
    model = 1;
end

if nargin < 8 || isempty(trate)
    trate = 100;
end

if nargin < 7 || isempty(nt)
    if ~isempty(tlist)
        nt = length(tlist);
    else
        nt = 100;
    end
end


tic();
[n,p] = size(X);
[~,p1] = size(X1);
beta = zeros(p,1);
z = zeros(p,1);
path = zeros(p,nt);

iter = 0;
if isempty(tlist)
    ind = (p1+1):p;
    while(true)
        iter = iter + 1;
        g = X'*grad(X*beta,y,model);
        z = z - g*(alpha/n);
        beta = shrinkage(z,p1)*kappa;
        index = (beta((p1+1):end)~=0);
        if any(index)
            t0 = iter*alpha + max(beta(ind(index))./g(ind(index))*(n/kappa));
            tlist = t0*trate.^((0:(nt-1))/(nt-1));
            break;
        end
    end
end
k = 1;
maxiter = ceil(tlist(nt)/alpha);
while(iter <= maxiter)
    dt = (iter*alpha-tlist(k));
    while (k<=nt && dt>0)
        path(:,k) = shrinkage(z+g*(dt/n),p1)*kappa;
        k = k+1;
        if (print)
            disp(strcat('Process:',num2str(100*iter/maxiter),'%, time cost ',num2str(toc()),' seconds.'));
        end
        if(k>nt)
            break;
        end
        dt = (iter*alpha-tlist(k));
    end
    iter = iter + 1;
    g = X'*grad(X*beta,y,model);
    z = z - g*(alpha/n);
    beta = shrinkage(z,p1)*kappa;
end
fit.path = path;
fit.tlist = tlist;
fit.alpha = alpha;
end

%------------------------------------------------------------------
% End function
%------------------------------------------------------------------

function X = shrinkage(z,p1)
X = sign(z).*max(abs(z)-[zeros(p1,1);ones(length(z)-p1,1)],0);
end
%------------------------------------------------------------------
% End function
%------------------------------------------------------------------

function g = grad(pred,y,model)
switch model
    case 1
        g = pred - y;
    case 2
        g = -y./(1+exp(y.*pred));
    case 3
        g = -y.*exp(-pred.^2/2)./normcdf(pred.*y)/sqrt(2*pi);
    otherwise
        error('Unsupported Model.');
end
end
%------------------------------------------------------------------
% End function
%------------------------------------------------------------------
