% 1. Load the dataset
data = readtable('data/HourlyDataset-SolarTechLab.csv');

% 2. Extract the date (day) from the HourlyTime column
data.Date = dateshift(data.HourlyTime, 'start', 'day'); % Extract only the date

% 3. Calculate daily statistics: mean and max of PV_Power
dailyStats = varfun(@(x) [mean(x), max(x)], data, 'InputVariables', 'PV_Power_Mean', ...
                    'GroupingVariables', 'Date', 'OutputFormat', 'table');
dailyStats.Properties.VariableNames{'Fun_PV_Power_Mean'} = 'Mean_Max'; % Rename output
dailyStats.Mean = dailyStats.Mean_Max(:, 1); % Extract mean
dailyStats.Max = dailyStats.Mean_Max(:, 2); % Extract max
dailyStats.Mean_Max = []; % Remove intermediate variable

% 4. Identify days where mean PV_Power < 10 or max PV_Power > 500
daysToRemove = dailyStats.Date(dailyStats.Mean < 10 | dailyStats.Max > 500);

% 5. Filter out rows corresponding to those days
cleanedData = data(~ismember(data.Date, daysToRemove), :);

% 6. Save the cleaned data to a new file
writetable(cleanedData, 'data/HourlyDataset-SolarTechLab-Filtered.csv');
