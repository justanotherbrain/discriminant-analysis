%% SVM vs FLDA
% How much data does the SVM and FLDA need in order to generalize.
% Separable and non-separable case.


%% Data
% We will create two datasets - one that's separable and another that's
% non-separable. We will create two distributions and draw samples from
% them.
rng(1) % for reproducibility
shuffle = randperm(20000);

theta = 60; %angle to rotate by
sXa = (randn(10000,2)*3 + 2) * [2 1; 1 1] * [cos(theta) -sin(theta); sin(theta) cos(theta)];
sXb = (randn(10000,2)*3) * [2 1; 1 1] * [cos(theta) -sin(theta); sin(theta) cos(theta)] + 8;
separable.X = [sXa; sXb]; % combine into single dataset
separable.Y = [zeros(10000,1);ones(10000,1)]; % set labels
s = separable; % for convenience
s.X = s.X(shuffle,:); % shuffle
s.Y = s.Y(shuffle,:);

nsXa = (randn(10000,2)*3 + 2) * [2 3; 1 1] * [cos(theta) -sin(theta); sin(theta) cos(theta)];
nsXb = (randn(10000,2)*3 + 2) * [2 3; 1 1] * [cos(theta) -sin(theta); sin(theta) cos(theta)] + 3;
nonseparable.X = [nsXa; nsXb]; % combine into single dataset
nonseparable.Y = [zeros(10000,1);ones(10000,1)]; % set labels
ns = nonseparable; % for convenience
ns.X = ns.X(shuffle,:); % shuffle
ns.Y = ns.Y(shuffle,:);

% Now we can plot the data
close all

figure()

subplot(1,2,1)
gscatter(s.X(:,1),s.X(:,2),s.Y,'rb')
title('Separable')

subplot(1,2,2)
gscatter(ns.X(:,1),ns.X(:,2),ns.Y,'rb')
title('Non-separable')

%% Set up validation set and test set for 10-fold cross validation

s.test.X = s.X(1:length(s.X)*.1, :); % leave 10% aside for test set
s.test.Y = s.Y(1:length(s.Y)*.1, :);
svalidX = s.X((length(s.Y)*.1+1):end,:);
svalidY = s.Y((length(s.Y)*.1+1):end,:);

ns.test.X = ns.X(1:length(ns.X)*.1, :);
ns.test.Y = ns.Y(1:length(ns.Y)*.1, :);
nsvalidX = ns.X((length(ns.X)*.1+1):end, :);
nsvalidY = ns.Y((length(ns.Y)*.1+1):end, :);

len = length(svalidX);
% set up 10-fold validation set
for i = 1:10
    s.valid.X{i} = svalidX(((i-1)*.1*len)+1:(i*.1*len),:);
    s.valid.Y{i} = svalidY(((i-1)*.1*len)+1:(i*.1*len),:);
    ns.valid.X{i} = nsvalidX(((i-1)*.1*len)+1:(i*.1*len),:);
    ns.valid.Y{i} = nsvalidY(((i-1)*.1*len)+1:(i*.1*len),:);
end


%% Don't run this section unless different distribution!
% With the artificial dataset, performance was best with C of .1, 1, or 10. It
% (obviously) didn't matter in the separable case. Since a higher C leads to a smaller margin,
% better generalization bounds are guaranteed with a lower C (larger margin) - therefore,
% we will stick with a C of .1.

% %% Get SVM parameter
% % This is a little tricky, since SVM has a parameter, while FLD doesn't.
% % Therefore, I will first plot performance of the SVM as a function of the
% % slack variable in both the separable and non-separable case using
% % grid-search.
% 
%C = [.001; .01; .1; 1; 10];
% 
% for i = 1:length(C)
%     spc = zeros(10,1);
%     nspc = zeros(10,1);
%     for j = 1:10
%         scrossX = [];
%         scrossY = [];
%         nscrossX = [];
%         nscrossY = [];
%         svalidX = s.valid.X{j};
%         svalidY = s.valid.Y{j};
%         nsvalidX = ns.valid.X{j};
%         nsvalidY = ns.valid.Y{j};
%         for k = 1:10
%             if j ~= k
%                 scrossX = [scrossX; s.valid.X{k}];
%                 scrossY = [scrossY; s.valid.Y{k}];
%                 nscrossX = [nscrossX; ns.valid.X{k}];
%                 nscrossY = [nscrossY; ns.valid.Y{k}];
%             end
%         end
%         % train model
%         smodel = fitcsvm(scrossX, scrossY, 'boxconstraint', C(i));
%         nsmodel = fitcsvm(nscrossX, nscrossY, 'boxconstraint', C(i));
%         
%         % test model
%         spred = predict(smodel, svalidX);
%         nspred = predict(nsmodel, nsvalidX);
%         
%         % calculate percent correct
%         spc(j) = sum(abs(spred - svalidY)) / length(spred);
%         nspc(j) = sum(abs(nspred - nsvalidY)) / length(nspred);
%     end
%     s.svm.cvparam{i}.C = C(i);
%     s.svm.cvparam{i}.mean = mean(spc);
%     s.svm.cvparam{i}.std = std(spc);
%     ns.svm.cvparam{i}.C = C(i);
%     ns.svm.cvparam{i}.mean = mean(nspc);
%     ns.svm.cvparam{i}.std = std(nspc);
% end
%  
% smeans = [];
% sstds = [];
% nsmeans = [];
% nsstds = [];
% for i = 1:length(ns.svm.cvparam)
%     smeans = [smeans; s.svm.cvparam{i}.mean];
%     sstds = [sstds; s.svm.cvparam{i}.std];
%     nsmeans = [nsmeans; ns.svm.cvparam{i}.mean];
%     nsstds = [nsstds; ns.svm.cvparam{i}.std];
% end
% 
% figure()
% 
% subplot(1,2,1)
% errorbar(C,smeans,sstds)subplot(1,2,1)
gscatter(s.test.X(:,1),s.test.X(:,2),s.test.Y,'rb')
title('Separable')

subplot(1,2,2)
gscatter(ns.test.X(:,1),ns.test.X(:,2),ns.test.Y,'rb')
title('Non-separable')

% title('Separable')
% 
% subplot(1,2,2)
% errorbar(C,nsmeans,nsstds)
% title('Non-separable')


%% Now train and test svm and fld with varying amounts of data
% Use varying amounts of data to train and test svm vs fld. 
% Need fewer points!!!

C = .1;
N = [10, 100, 1000];
svm_spc = zeros(10,1);
svm_nspc = zeros(10,1);
fld_spc = zeros(10,1);
fld_nspc = zeros(10,1);
for i = 1:length(N)
    sX = [];
    sY = [];
    nsX = [];
    nsY = [];
    for j = 1:i
        sX = [sX; s.valid.X{j}];
        sY = [sY; s.valid.Y{j}];
        nsX = [nsX; ns.valid.X{j}];
        nsY = [nsY; ns.valid.Y{j}];
    end
    sX = s.valid.X{1}(1:N(i),:);
    sY = s.valid.Y{1}(1:N(i),:);
    nsX = ns.valid.X{1}(1:N(i),:);
    nsY = ns.valid.Y{1}(1:N(i),:);


    % train svm
    smodel = fitcsvm(sX, sY, 'boxconstraint', C);
    nsmodel = fitcsvm(nsX, nsY, 'boxconstraint', C);

    % test svm
    spred = predict(smodel, s.test.X);
    nspred = predict(nsmodel, ns.test.X);

    % calculate percent correct
    svm_spc(i) = sum(abs(spred - s.test.Y)) / length(spred);
    svm_nspc(i) = sum(abs(nspred - ns.test.Y)) / length(nspred);
    
%     train fld
%     [sw, st, ~] = fisher_training(sX, sY);
%     [nsw, nst, ~] = fisher_training(nsX, nsY);
%     
%     test fld
%     sp = fisher_testing(s.test.X, sw, st);
%     nsp = fisher_testing(ns.test.X, nsw, nst);
    
    sparams = TrainFLDA(sX,sY);
    nsparams = TrainFLDA(nsX, nsY);
    
    sp = TestFLDA(sparams, s.test.X, [1,0]);
    nsp = TestFLDA(nsparams, ns.test.X, [1,0]);
    
    % calculate percent correct
    fld_spc(i) = sum(abs(sp - s.test.Y)) / length(sp);
    fld_nspc(i) = sum(abs(nsp - ns.test.Y)) / length(nsp);
    
    
end

%%

figure
plot((1:10),[svm_spc'; svm_nspc'; fld_spc'; fld_nspc'])
legend('svm s','svm ns','fld s','fld ns')
title('Error rate per number of training samples')
xlabel('number of samples')
ylabel('error rate')
