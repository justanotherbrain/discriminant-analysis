function params=TrainQDA(X,y,lambda)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TrainQDA
% Trains quadratic discriminant analysis
%
% X is the data matrix, rows are examples
% y are the responses
% lambda sets regularization [0,1] (default 0)
%
% rabadi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 3
    lambda = 0;
end

labels = unique(y);

X0 = X(y==labels(1),:);
X1 = X(y==labels(2),:);

mu0 = mean(X0)';
mu1 = mean(X1)';


C0 = cov(X0);
C1 = cov(X1);

Q = -.5*(inv(C0)-inv(C1));

m = C0\mu0 - C1\mu1;

k = -0.5*(log(det(C0)) - log(det(C1)) + mu0'*inv(C0)*mu0 - mu1'*inv(C1)*mu1);

params.Q = Q;
params.m = m;
params.k = k;

end