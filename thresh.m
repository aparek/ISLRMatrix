function [ y ] = thresh( x, lam, a, pen, p )
%y = thresh( x, lam, a, pen, p )
%
% This function computes generalized threshold for
% variety of penaly functions. 
%
% Input: 
%       x - input signal
%       lam  - regularizer lambda
%       a - nonconvex penalty parameter, see [1]
%       pen - type of penalty [firm, log, atan, l1, lp]
%       p - p value for lp penalty see [2]
%
% Output:
%       y - thresholded signal 
% 
% Please cite as [1] below. 
%
% Contact: Ankit Parekh, ankit.parekh@nyu.edu
% Last Edit: 11/24/16.
% [1]. Improved Sparse and Low-Rank Matrix Estimation. (PrePrint)
%      A. Parekh and I. W. Selesnick. Preprint https://arxiv.org/abs/1605.00042 
% 
% [2]. Compressed sensing recovery via nonconvex shrinkage penalties
%      J. Woodworth and R. Chartrand
%      Inverse Problems 32 075004, 2016. 


y = zeros(size(x));
ind = abs(x)>=lam;

if nargin == 4
    p = 0;
end
if ~lam
    y = x;
else

    switch pen
        case 'firm'
            y = ( 1/(1-a*lam)*max(abs(x)-lam,0) + (1-1/(1-a*lam))*max(abs(x)-1/a,0) ).*sign(x);
        case 'log'
            y(ind) = (abs(x(ind))/2-1/(2*a) + sqrt((abs(x(ind))./2 + 1/(2*a)).^2 - lam/a)).*sign(x(ind));
        case 'atan'
            y = atanT(x,lam,a);
        case 'l1'
            y = soft(x,lam);
        case 'lp'
            y(ind) = max((abs(x(ind)) - lam.^(2-p).*abs(x(ind)).^(p-1)),0).*sign(x(ind));
        otherwise
            disp('Please select penalty from the following: log, atan, l1')
    end
end


