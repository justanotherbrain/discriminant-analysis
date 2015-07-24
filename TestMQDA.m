function yhat = TestMQDA(params,X,vals)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TestMQDA
% Test the mixture QDA learned from TrainMQDA
%
% params are parameters returned by TrainMQDA
% vals is the output variables (default [-1,1])
% yhat is estimated labels
%
% rabadi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 3
    vals = [-1,1];
end

n = length(params);

yhat = zeros(size(X,1),1);


for i = 1:length(yhat)
    temp = 0;
    for t = 1:n
        temp = temp + abs(params{t}.alpha) * (X(i,:)*params{t}.Q*X(i,:)' + params{t}.m'*X(i,:)' + params{t}.k);
    end
    test = sign(temp);
    if test == -1
        yhat(i) = vals(1);
    elseif test == 1
        yhat(i) = vals(2);
    else
        yhat(i) = vals(randi((1:2)));
    end

end
end