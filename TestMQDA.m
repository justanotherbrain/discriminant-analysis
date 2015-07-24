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

m = zeros(n,2);
for t = 1:n
    m(t,:) = params{t}.m;
end
for i = 1:length(yhat)
    
    % for AdaBoost
%     temp = 0;
%     for t = 1:n
%         temp = temp + abs(params{t}.alpha) * (X(i,:)*params{t}.Q*X(i,:)' + params{t}.m'*X(i,:)' + params{t}.k);
%     end

    % proximity to m
    I = GI(X(i,:),m);
    temp = X(i,:)*params{I}.Q*X(i,:)' + params{I}.m'*X(i,:)' + params{I}.k;
    
    
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

function I = GI(x,m)
    dists = zeros(size(m,1),1);
    for i = 1:size(m,1)
        dists(i) = pdist([x; m(i,:)]);
    end
    [~, I] = min(dists);
end