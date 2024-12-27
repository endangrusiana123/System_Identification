close all;

% Load data from CSV file
%data = readtable('data/HourlyDataset-Filled.csv');
%data = readtable('data/CleanedDataset.csv');
data = readtable('data/CleanedDataset_Filtered.csv');

% Convert Time column to datetime
data.HourlyTime = datetime(data.HourlyTime, 'InputFormat', 'dd-MMM-yyyy HH:mm:ss');

% Number of columns in the table
num_columns = width(data);


% Loop through each column and plot in a new figure
for i = 2:num_columns % Skip the first column if it's 'Time'
    column_name = data.Properties.VariableNames{i};
    figure; % Create a new figure for each column
    plot(data.HourlyTime, data{:, i}, 'LineWidth', 1.5);
    grid on;
    title(['Plot of ', column_name], 'Interpreter', 'none');
    xlabel('Time');
    ylabel(column_name);

    % Save the figure
    saveas(gcf, ['result/dataset_' column_name '.png']);
end
