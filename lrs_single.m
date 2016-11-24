function [X, cost] = lrs_single(Y, k, lam0, lam1, mu, pen, Nit,p)
% [X, cost] = lrs_single(Y, k, lam0, lam1, mu, pen, Nit)
%     This function estimates the matrix X, which is sparse and low-rank,
%     from the noisy input matrix Y. 
%     
%     Input:
%         Y - Noisy matrix
%         k - Constant for selecting a0 along the line. (0 < k < 1)
%         lam0 - Regularization parameter for Singular value penalty
%         lam1 - Regularization parameter for Sparse penalty
%         mu - ADMM parameters (step sizes). 
%         pen - Penalty function ('L1','atan','log')
%         Nit - Number of iterations
%         
%     Output:
%         X - Estimated sparse and low-rank matrix 
%         cost - Cost function history
%         
% Last Edit: 31st Aug, 2015.
% Contact: Ankit Parekh ankit.parekh@nyu.edu
%
% Please cite as: 
% Improved Sparse and Low-Rank Matrix Estimation. (PrePrint)
% A. Parekh and I. W. Selesnick. Preprint https://arxiv.org/abs/1605.00042 

switch pen
    case 'log'
        phi = @(x, a) (1/a) * log10(1 + a*abs(x));                    
    case 'atan'
        phi = @(x, a) 2./(a*sqrt(3)) .* (atan((2*a.*abs(x)+1)/sqrt(3)) - pi/6);
    case 'l1'
        phi = @(x,a) abs(x);
    case 'lp'
        phi = @(x,p) abs(x).^p;
    case 'firm'
        phi = @(x,a) zeros(size(x)) +...                                            
             (abs(x) < (1/a)).* (abs(x) - (a/2)*x.^2) + ...
             (abs(x) >= (1/a)) .* (1/(2*a));
end

if nargin < 8
    p = 0;
end

if lam0
    a0 = k/lam0;
else
    a0 = 0;
end

if lam1
    a1 = (1-a0*lam0) / (lam1);
else
    a1 = 0;
end

X = zeros(size(Y));
D = X;
U = X;
cost = zeros(Nit,1);
alpha = 1/(1+mu);

for i = 1:Nit
    %X-step
    if a1 == 0
        X = thresh(alpha * (Y + mu * (U + D)),(lam1*alpha),a1,'l1',1);
    else
        X = thresh(alpha * (Y + mu * (U + D)),(lam1*alpha),a1,pen,p);
    end
    
    
    %U-step
    [P,Sigma,Q] = svd(X-D,'econ');
    if a0 == 0
        U = P * diag(thresh(diag(Sigma),lam0/mu,a0,'l1',1)) * Q';
    else
        U = P * diag(thresh(diag(Sigma),lam0/mu,a0,pen,p)) * Q';
    end
    
    %D-step
    D = D - (X-U);
      
    %Calculate cost function history
    if strcmp(pen,'lp')
       cost(i) = 0.5*norm(Y-X,'fro')^2 + ... 
                    lam0 * sum(phi(svd(X,'econ'),p)) + ...
                    lam1 * sum(sum(phi(X,p),1));
    else
        cost(i) =  0.5*norm(Y-X,'fro')^2 + ...
                               lam0*sum(phi(svd(X,'econ'),a0)) + ...
                             + lam1*sum(sum(phi(X,a1),1));
    end
                         
end
if issparse(Y)
    X = sparse(X);
end

end

