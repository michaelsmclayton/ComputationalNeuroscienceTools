
%% Load data
dataDirectory = './data/';
categories = load([dataDirectory, 'stimuli_class_assignment_confident.mat']);
neuralDataFiles = dir([dataDirectory, 'natimg2800_M1*.mat']);

%% Loop over data files
spontaneousActivity = [];
correlationByGroups = cell(1,length(unique(categories.class_assignment)));
for currentDataFile = 1:length(neuralDataFiles)
    
    % Load data
    dataPath = [neuralDataFiles(currentDataFile).folder, '/', neuralDataFiles(currentDataFile).name];
    disp(['Loading ', dataPath])
    load(dataPath)
    
    % Get correlation matrix
    Corr = corr(stim.resp');
    
    % Spontaneous correlation
    sponCorr = corr(stim.spont');
    sponCorrArray = reshape(sponCorr, [1, size(sponCorr,1)*size(sponCorr,2)]);
    sponCorrArray(sponCorrArray==1) = [];
    spontaneousActivity = horzcat(sponCorrArray);
    
    % Make sure that image numbers are less than 2800
    stim.istim = stim.istim(stim.istim <= 2800);
    
    % Loop over images
    for imageNumber = unique(stim.istim)'
    
        % Get correlation between activity for two images
        imageIndices = find(stim.istim==imageNumber);
        if length(imageIndices)==2
            currentCorrelation = Corr(imageIndices(1), imageIndices(2));
        end

        % Add correlation to current class
        currentClass = categories.class_assignment(imageNumber)+1;
        correlationByGroups{currentClass} = horzcat(correlationByGroups{currentClass}, ...
            currentCorrelation);
    
    end
end


%% Plot results (i.e. correlations across image types)
% figure(); hold on
figure('Position', [0 0 1100 600]); hold on

% Add natural images scatter
for currentClass = 1:length(correlationByGroups)
    y = correlationByGroups{currentClass};
    x = currentClass*ones(size(y));
    scatter(x,y)
    scatter(currentClass, mean(y), 100, 'd', 'MarkerFaceColor', [1 1 1], 'LineWidth', 3, 'MarkerEdgeColor', [0 0 0])
    [h,p,c] = ttest(y);
    disp([categories.class_names(currentClass), ' = ', num2str(p)])
end

% Add spontaneous scatter
y = spontaneousActivity;
x = (length(correlationByGroups)+2)*ones(size(y));
scatter(x, y, 'MarkerEdgeColor', [.5 .5 .5])
scatter(length(correlationByGroups)+2, mean(y), 100, 'd', 'MarkerFaceColor', [1 1 1], 'LineWidth', 3, 'MarkerEdgeColor', [0 0 0])
xlim([0, length(correlationByGroups)+3])

xticks(horzcat(1:length(correlationByGroups), length(correlationByGroups)+2))
xticklabels(horzcat(categories.class_names, 'spontaneous'))

set(gcf, 'Color', 'w')
set(gca, 'FontSize', 13)
set(gca, 'LineWidth', 2)
ylabel('Correlation coefficient')


% %% Show examples of correlation
% %close all
% figure();
% lineWidth = 2;
% s1 = subplot(3,1,1);
% s1.Position = s1.Position + [0 0 0 -.1];
% sin1 = stim.resp(1,:);
% sin2 = stim.resp(2,:);
% plot(sin1, 'LineWidth', lineWidth)
% ylabel('Activity'); xlabel('Neuron')
% % box off; axis off
% xlim([1 length(sin1)])
% s2 = subplot(3,1,2);
% s2.Position = s2.Position + [0 +.1 0 -.1];
% plot(sin1, 'LineWidth', lineWidth)
% ylabel('Activity'); xlabel('Neuron')
% % box off; axis off
% xlim([1 length(sin1)])
% s3 = subplot(3,1,3);
% s3.Position = s3.Position + [0 0 0 +.1];
% scatter(sin2, sin1, 2)
% set(gcf, 'Color', 'w')
% set(gca, 'LineWidth', 2)
% %xticks([-1, 0, 1])
% %yticks([-1, 0, 1])