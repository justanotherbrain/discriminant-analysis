function params=TrainMQDA(X,y,n1, n2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TrainMQDA
% Trains mixture quadratic discriminant analysis
%
% X is the data matrix, rows are examples
% y are the responses
% lambda sets regularization [0,1] (default 0)
%
% rabadi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
warning('off','stats:gmdistribution:FailedToConverge')

%%
labels = unique(y);

X0 = X(y==labels(1),:);
X1 = X(y==labels(2),:);

model0 = gmdistribution.fit(X0, n1);
model1 = gmdistribution.fit(X1, n2);

%% For just one combination
% for i = 1:n
% 
%     mu0 = model0.mu(i,:)';
%     mu1 = model1.mu(i,:)';
% 
%     C0 = model0.Sigma(:,:,i);
%     C1 = model1.Sigma(:,:,i);
% 
%     Q = -.5*(inv(C0)-inv(C1));
% 
%     m = C0\mu0 - C1\mu1;
% 
%     k = -0.5*(log(det(C0)) - log(det(C1)) + mu0'*inv(C0)*mu0 - mu1'*inv(C1)*mu1);
% 
%     params{i}.Q = Q;
%     params{i}.m = m;
%     params{i}.k = k;
% end

%% For all pairwise combinations between distributions (n^2)
iter=0;
for i = 1:n1
    for j = 1:n2
        mu0 = model0.mu(i,:)';
        mu1 = model1.mu(j,:)';
        
        C0 = model0.Sigma(:,:,i);
        C1 = model1.Sigma(:,:,j);
        
        Q = -.5*(inv(C0)-inv(C1));

        m = C0\mu0 - C1\mu1;

        k = -0.5*(log(det(C0)) - log(det(C1)) + mu0'*inv(C0)*mu0 - mu1'*inv(C1)*mu1);
        
        iter = iter+1;
        params{iter}.Q = Q;
        params{iter}.m = m;
        params{iter}.k = k;
    end
end

%% AdaBoost

% D = ones(length(y),1) * 1/length(y);
% 
% yd = y;
% yd(yd==0)=-1;
% 
% if iter > 1
%     for t = 1:iter
%         prediction = TestQDA(params{t}, X, [labels(2) labels(1)]);
%         errors = sum((D .* abs(prediction - y)));
%         if errors == 0
%             errors = 0.000000001; % avoid divide by zero
%         end
%         alpha = 0.5 * log((1-errors)/errors);
%         Z = 2 * (errors*(1-errors))^.5;
% 
%         for i = 1:length(y)
%             D(i) = abs((D(i) * exp(-alpha*yd(i)*prediction(i))) / Z);
%         end
%         params{t}.alpha = abs(alpha);
%     end
% else
%     params{1}.alpha = 1;
% end

%% Find best combination - go with closest means
% for t = 1:iter
%     
% end



end