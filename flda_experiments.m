%% Remove stupid warning message

warning('off','MATLAB:colon:nonIntegerIndex');

%% Clean stuff up
close all
clc
%% DATA

rng(1) % for reproducibility
shuffle = randperm(2000);


%% Clearly not the best hyperplane
% theta1 = 45; %angle to rotate by
% theta2 = 15;
% 
% Xa = (randn(1000,2)*3 + 10) * [2 1; 1 2] * [cos(theta1) -sin(theta1); sin(theta1) cos(theta1)];
% Xb = (randn(1000,2)*3) * [2 1; 1 3] * [cos(theta2) -sin(theta2); sin(theta2) cos(theta2)] + 0;

%% This works pretty well
% 
% theta1 = 45; %angle to rotate by
% theta2 = 90;
% 
% Xa = (randn(1000,2)) * [2 1; 1 2] * [cos(theta1) -sin(theta1); sin(theta1) cos(theta1)];
% Xb = (randn(1000,2)) * [2 1; 1 3] * [cos(theta2) -sin(theta2); sin(theta2) cos(theta2)] + 10;

%% The LDA is so bad! Surprising, really.

theta1 = 45; %angle to rotate by
theta2 = 0;

Xa = (randn(1000,2)) * [2 1; 1 2] * [cos(theta1) -sin(theta1); sin(theta1) cos(theta1)];
Xb = (randn(1000,2)) * [2 1; 1 3] * [cos(theta2) -sin(theta2); sin(theta2) cos(theta2)] + 10;
Xb(:,1) = Xb(:,1)-10;

%% Now for something a little more tricky - In this case, the QDA fails, but LDA is standing strong
% 
% theta1 = 45;
% theta2 = 180;
% 
% Xa = randn(1000,2) * [cos(theta1) -sin(theta1); sin(theta1) cos(theta1)];
% Xb1 = randn(1000,1);
% Xb2 = .3*Xb1.^2 + randn(1000,1)*.1;
% Xb = [Xb1, Xb2];
% Xb = Xb * [cos(theta2) -sin(theta2); sin(theta2) cos(theta2)];
% Xb(:,1) = Xb(:,1)+5;
% Xb(:,2) = Xb(:,2)+3;


%% Do the rest

X = [Xa; Xb]; % combine into single dataset
Y = [zeros(1000,1);ones(1000,1)]; % set labels

X = X(shuffle,:); % shuffle
Y = Y(shuffle,:);

figure()

gscatter(X(:,1),X(:,2),Y,'rb')
title('Separable')

C = .1;
N = [10, 50, 100, 150];

test.X = X(1:length(X)*.1, :); % leave 10% aside for test set
test.Y = Y(1:length(Y)*.1, :);
validX = X((length(Y)*.1+1):end,:);
validY = Y((length(Y)*.1+1):end,:);

len = length(validX);

fld_pc = zeros(length(N),1);
qda_pc = zeros(length(N),1);

all_lda_params = [];
all_qda_params = [];
for i = 1:length(N)
    
    sX = validX(1:N(i),:);
    sY = validY(1:N(i),:);

    
    lda_params = TrainFLDA(sX,sY);
    qda_params = TrainQDA(sX,sY);
    
    all_lda_params = [all_lda_params, lda_params];
    all_qda_params = [all_qda_params, qda_params];
    
    lda_p = TestFLDA(lda_params, test.X, [1,0]);
    qda_p = TestQDA(qda_params, test.X, [1,0]);
    
    % calculate percent correct
    fld_pc(i) = sum(abs(lda_p - test.Y)) / length(lda_p); 
    qda_pc(i) = sum(abs(qda_p - test.Y)) / length(qda_p);
end

figure
plot((N),[fld_pc'; qda_pc'])
legend('lda', 'qda')
title('FLDA Error rate per number of training samples')
xlabel('number of samples')
ylabel('error rate')

VisualizeFLDA(all_lda_params,test.X,test.Y);
VisualizeQDA(all_qda_params,test.X,test.Y);

% VisualizeFLDA(all_lda_params,X,Y);
% VisualizeQDA(all_qda_params,X,Y);

% VisualizeFLDA(all_lda_params,sX,sY);
% VisualizeQDA(all_qda_params,sX,sY);