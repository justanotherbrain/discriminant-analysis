function VisualizeFLDA(params,data,labels)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%s
% VisualizeFLDA
% Visualize the fisher linear discriminate analysis with parameters from
% TrainFLDA. Only works in two dimensions
%
% params - parameters learned with TrainFLDA
% data - data to plot
%
% rabadi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w = params.W;
k = params.k;

a = -w(1)/w(2);

xx = linspace(-50,50);


% get hyperplane
yy = a * xx  - k/w(2); %(k/w(1)); 

    
figure
hold on
gscatter(data(:,1),data(:,2),labels,'rb')
title('Visualize Fisher')
plot(xx,yy,'k-')
hold off
