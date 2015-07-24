function VisualizeMQDA(params,data,labels)
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

warning('off','MATLAB:ezplotfeval:NotVectorized')

n = size(params,2);
if size(params,1)==1
    Q11 = zeros(n,1);
    Q12 = zeros(n,1);
    Q21 = zeros(n,1);
    Q22 = zeros(n,1);
    m1 = zeros(n,1);
    m2 = zeros(n,1);
    k = zeros(n,1);
    alpha = zeros(n,1);
    for t = 1:n
        Q = params{t}.Q;
        Q11(t) = Q(1,1);
        Q12(t) = Q(1,2);
        Q21(t) = Q(2,1);
        Q22(t) = Q(2,2);
        m = params{t}.m;
        m1(t) = m(1);
        m2(t) = m(2);
        k(t) = params{t}.k;
        alpha(t) = params{t}.alpha;
    end
    xx = linspace(-50,50);

    % get hyperplane
    %yy =  xx * (a' * a) * xx' + b * xx' + c;  

    f = @(x1,x2) sum(alpha .* ( k + m1.*x1 + m2.*x2 + Q11.*x1.^2 + ...
        (Q12+Q21).*x1.*x2 + Q22.*x2.^2));

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
    
    for i = 1:size(params,1)
        n = size(params,2);
        Q11 = zeros(n,1);
        Q12 = zeros(n,1);
        Q21 = zeros(n,1);
        Q22 = zeros(n,1);
        m1 = zeros(n,1);
        m2 = zeros(n,1);
        k = zeros(n,1);
        alpha = zeros(n,1);
        for t = 1:n
            Q = params{i,t}.Q;
            Q11(t) = Q(1,1);
            Q12(t) = Q(1,2);
            Q21(t) = Q(2,1);
            Q22(t) = Q(2,2);
            m = params{i,t}.m;
            m1(t) = m(1);
            m2(t) = m(2);
            k(t) = params{i,t}.k;
            alpha(t) = params{i,t}.alpha;
        end

        xx = linspace(-50,50);

        % get hyperplane
        %yy =  xx * (a' * a) * xx' + b * xx' + c;  

        f = @(x1,x2) sum(alpha .* ( k + m1.*x1 + m2.*x2 + Q11.*x1.^2 + ...
            (Q12+Q21).*x1.*x2 + Q22.*x2.^2));

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

