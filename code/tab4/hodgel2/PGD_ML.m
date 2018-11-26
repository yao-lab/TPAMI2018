function [M_path,lambda,info] = PGD_ML(A,y,p1,p2,svd_top,LS,lambda,nlambda,lambda_rate,gamma)
if nargin < 5
    svd_top = 1;
end
if nargin < 6
    LS = 1;
end

if LS
    ng = reshape(A'*y,p1,p2);
else
    ng = reshape(A'*(y-1/2),p1,p2);
end
lambda_max = svds(ng,1,'L');

if nargin < 7 || isempty(lambda)
    rho=lambda_rate^(1/(nlambda-1));
    lambda = lambda_max*rho.^[0:(nlambda-1)];
end
nlambda = length(lambda);
if nargin < 10
    ss = svds(A,1,'L');
    if LS
        gamma = ss^-2;
    else
        gamma = 4*ss^-2;
    end
end

maxIter = 1000;
eps1 = 1e-4; % relative error to stop
M = zeros(p1,p2);

step_diff = [];
minp = min(p1,p2);
k = 2;

m0 = sum(lambda>=lambda_max);
M_path = zeros(p1*p2,nlambda);
for i = 1:m0
    info.count(i) = 0;
    info.rank(i) = 0;
end
for i = (m0+1):length(lambda)
    count = 0;
    thresh = lambda(i)*gamma;
    while count < maxIter
        count = count+1;
        if LS
            ng = reshape(A'*(y - A*M(:)),p1,p2);
        else
            ng = reshape(A'*(y - 1./(1+exp(-A*M(:)))),p1,p2);
        end
        temp = M + gamma*ng;
        if svd_top
            while (1)
                [u,s,v] = svds(temp,k,'L');
                if (sum(diag(s)>thresh)<k || k ==minp)
                    break
                else
                    k = min(k+2,minp);
                end
            end
        else
            [u,s,v] = svd(temp);
        end
        M_new = u*max(0,s-thresh)*v';
        k = sum(diag(s)>thresh)+1;
        
        step_diff(count) = norm(M_new - M,'fro')/(norm(M,'fro')+eps);
%         if (mod(count,100)==0)
%             count
%             step_diff(count)
%         end
        M = M_new;
        if (step_diff(count) < eps1)
            M_path(:,i) = M(:);
            info.count(i) = count;
            info.rank(i) = k-1;
            break
        end
    end
end