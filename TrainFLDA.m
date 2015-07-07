function params = TrainFLDA(X,y,lambda)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TrainFLDA
% Train a fisher linear discriminant analysis
%
% X is the data matrix (each row is data point)
% y is vector of labels, binary labels {-1,1}
% lambda - [0,1) regularizes estimate of the covariance matrix against the
% identity matrix
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

C = (1-lambda)*0.5*(C0+C1) + lambda*eye(size(C0)); % average covariance matrix


W = C\(mu0-mu1);

k = 0.5*((mu1'*inv(C)*mu1) - (mu0'*inv(C)*mu0));


params.W = W;
params.k = k;
end