function yhat = TestQDA(params,X,vals)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TestQDA
% Test the QDA learned from TrainFLDA
%
% params are parameters returned by TrainFLDA
% vals is the output variables (default [-1,1])
% yhat is estimated labels
%
% rabadi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 3
    vals = [-1,1];
end

yhat = zeros(size(X,1),1);

for i = 1:length(yhat)
    temp = sign(X(i,:)*params.Q*X(i,:)' + params.m'*X(i,:)' + params.k);

    if temp == -1
        yhat(i) = vals(1);
    elseif temp == 1
        yhat(i) = vals(2);
    else
        yhat(i) = vals(randi((1:2)));
    end
        
end