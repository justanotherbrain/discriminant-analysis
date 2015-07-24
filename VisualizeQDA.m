function VisualizeQDA(params,data,labels)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%s
% VisualizeQDA
% Visualize the quadratic discriminate analysis with parameters from
% TrainQDA. Only works in two dimensions
%
% params - parameters learned with TrainQDA
% data - data to plot
%
% rabadi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if length(params)==1

    Q = params.Q;
    m = params.m;
    k = params.k;

    xx = linspace(-50,50);

    % get hyperplane
    %yy =  xx * (a' * a) * xx' + b * xx' + c;  

    f = @(x1,x2) k + m(1)*x1 + m(2)*x2 + Q(1,1)*x1.^2 + ...
        (Q(1,2)+Q(2,1))*x1.*x2 + Q(2,2)*x2.^2;

    figure
    hold on
    gscatter(data(:,1),data(:,2),labels,'rb')
    %h = ezplot(f,[-40 40 -15 50]);
    h = ezplot(f,[min(data(:,1)) max(data(:,1)) min(data(:,2)) max(data(:,2))]);
    h.Color = 'k';
    h.LineWidth = 1;
    xlabel('');
    ylabel('');
    hold off
    title('Visualize QDA')
    
else

    figure
    for i = 1:length(params)
        Q = params(i).Q;
        m = params(i).m;
        k = params(i).k;

        xx = linspace(-50,50);

        % get hyperplane
        %yy =  xx * (a' * a) * xx' + b * xx' + c;  

        f = @(x1,x2) k + m(1)*x1 + m(2)*x2 + Q(1,1)*x1.^2 + ...
            (Q(1,2)+Q(2,1))*x1.*x2 + Q(2,2)*x2.^2;

        subplot(ceil(length(params)/2),2,i)
        hold on
        gscatter(data(:,1),data(:,2),labels,'rb')
        h = ezplot(f,[min(data(:,1)) max(data(:,1)) min(data(:,2)) max(data(:,2))]);
        h.Color = 'k';
        h.LineWidth = 1;
        xlabel('');
        ylabel('');
        hold off
        title(['QDA entry: ' num2str(i)])
        
    end
end

end

