% Load data from CSV file
data = readtable('data/Dataset-SolarTechLab.csv');

% Convert the 'Time' column to datetime format
data.Time = datetime(data.Time, 'InputFormat', 'dd-MMM-yyyy HH:mm:ss');
% Menggunakan nilai 0 untuk mengganti NaN di kolom numerik
data.PV_Power = fillmissing(data.PV_Power, 'constant', 0);
data.T_air = fillmissing(data.T_air, 'constant', 0);
data.G_h = fillmissing(data.G_h, 'constant', 0);
data.G_tilt = fillmissing(data.G_tilt, 'constant', 0);
data.W_s = fillmissing(data.W_s, 'constant', 0);
data.W_d = fillmissing(data.W_d, 'constant', 0);
% Create a new column for hourly timestamps
data.HourlyTime = dateshift(data.Time, 'start', 'hour');

% Calculate the mean of all variables grouped by the hourly time
hourlyData = varfun(@mean, data, 'GroupingVariables', 'HourlyTime', ...
                    'InputVariables', {'PV_Power', 'T_air', 'G_h', 'G_tilt', 'W_s', 'W_d'});

% Rename the columns for easier interpretation
hourlyData.Properties.VariableNames = {'HourlyTime', 'GroupCount', ...
                                       'PV_Power_Mean', 'T_air_Mean', ...
                                       'G_h_Mean', 'G_tilt_Mean', ...
                                       'W_s_Mean', 'W_d_Mean'};

% Save the hourly data to a new CSV file
writetable(hourlyData, 'data/HourlyDataset-SolarTechLab.csv');

% Display the first few rows of the hourly data
disp(head(hourlyData));
