% Close all figures
close all;

% Load data from CSV
data = readtable('data/CleanedDataset_Filtered.csv');

hourInt = data.HourInteger;
pv_power = data.PV_Power_Filtered;  % PV Power Output
temperature = data.T_air_Mean;      % Input: Air Temperature
irradiance = data.G_h_Filtered;     % Input: Global Horizontal Irradiance
tilt = data.G_tilt_Filtered;        % Input: Global Tilt Irradiance
wind_speed = data.W_s_Filtered;     % Input: Wind Speed
wind_direction = data.W_d_Filtered; % Input: Wind Direction
% Combine inputs into a matrix
inputs = [hourInt pv_power, irradiance, temperature, tilt, wind_speed, wind_direction];

% Shift PV power forward by 24 rows
shift = 24; % Number of rows to shift
output = [pv_power(shift+1:end); NaN(shift, 1)]; % Add NaN for shifted rows

% Truncate inputs to match output length
inputs = inputs(1:end-shift, :); % Remove last 'shift' rows from inputs
% time = time(1:end-shift);        % Adjust time to match inputs length

% Remove NaN rows from output
validIdx = ~isnan(output);
X_input = inputs(validIdx, :);
Y_output = output(validIdx);
% time = time(validIdx);

disp(['inputs size: ', num2str(size(X_input))])
disp(['output size: ', num2str(size(Y_output))])

% Split data into sample data and test sets
split_ratio = 0.6; % 80% for sistem identifying, 20% for testing
n_sample = floor(split_ratio * size(X_input, 1)); % Correct size for sample data

sample_data = iddata(Y_output(1:n_sample, :), X_input(1:n_sample, :), 1);
test_data = iddata(Y_output(n_sample+1:end, :), X_input(n_sample+1:end, :), 1);
%test_data.InputData

% Initialize results table
results = table();
% Create and identify ARX model
% Number of hours per day
hours_per_day = 24;
% na_uji = [0, 1, 2, 4, 8];
nb_uji = [1, 4, 8 , 24];
% for i = 1:length(na_uji)
for i = 1:length(nb_uji)
    input_day = 1;
    nValue = nb_uji(i);%input_day * hours_per_day; % Number of previous days as input
    na = 23;%na_uji(i); % Order of output delay
    nb = [nValue, nValue, nValue, nValue, nValue, nValue, nValue]; % Order for each input (irradiance, temperature, tilt, wind speed, wind direction)
    %nk = [hours_per_day, hours_per_day, hours_per_day, hours_per_day, hours_per_day,hours_per_day]; % Delay between input and output for each input
    nk = [0,0,0,0,0,0,0];
    model_arx = arx(sample_data, [na nb nk]);
    
    % Validate the model using sample data
    y_sample_pred = predict(model_arx, sample_data);
    
    % Validate the model using test data
    y_test_pred = predict(model_arx, test_data);
    
    % Evaluate model performance on sample data
    y_sample_actual = sample_data.OutputData;
    y_sample_forecast = y_sample_pred.OutputData;
    
    mae_sample = mean(abs(y_sample_actual - y_sample_forecast));
    rmse_sample = sqrt(mean((y_sample_actual - y_sample_forecast).^2));
    
    % Calculate FIT for sample data
    numerator_sample = norm(y_sample_actual - y_sample_forecast);
    denominator_sample = norm(y_sample_actual - mean(y_sample_actual));
    FIT_sample = 100 * (1 - numerator_sample / denominator_sample);
    
    % Display evaluation results for sample data
    % disp(['na value: ', num2str(na_uji(i))]);
    disp(['nb value: ', num2str(nb_uji(i))]);
    disp(['ARX Identify Mean Absolute Error (MAE): ', num2str(mae_sample)]);
    disp(['ARX Identify Root Mean Square Error (RMSE): ', num2str(rmse_sample)]);
    disp(['ARX Identify Forecasting Index Test (FIT): ', num2str(FIT_sample), '%']);
    
    % Evaluate model performance on test data
    y_test_actual = test_data.OutputData;
    y_test_forecast = y_test_pred.OutputData;
    mae_test = mean(abs(y_test_actual - y_test_forecast));
    rmse_test = sqrt(mean((y_test_actual - y_test_forecast).^2));
    
    % Calculate FIT for test data
    numerator_test = norm(y_test_actual - y_test_forecast);
    denominator_test = norm(y_test_actual - mean(y_test_actual));
    FIT_test = 100 * (1 - numerator_test / denominator_test);
    
    % Display evaluation results for test data
    disp(['Test Mean Absolute Error (MAE): ', num2str(mae_test)]);
    disp(['Test Root Mean Square Error (RMSE): ', num2str(rmse_test)]);
    disp(['Test Forecasting Index Test (FIT): ', num2str(FIT_test), '%']);
    
    % Append to results table
    % results = [results; table(na_uji(i), mae_sample, rmse_sample, FIT_sample, mae_test, rmse_test, FIT_test)];
    results = [results; table(nb_uji(i), mae_sample, rmse_sample, FIT_sample, mae_test, rmse_test, FIT_test)];
    
    % Visualize the prediction results
    figure;
    subplot(2,1,1);
    plot(sample_data.OutputData, 'b'); hold on;
    plot(y_sample_pred.OutputData, 'r--'); hold off;
    legend('Actual (Train)', 'Predicted (Train)');
    xlabel('Time (hours)');
    ylabel('PV Power');
    % title('ARX Model Validation for na value -', num2str(na_uji(i)));
    % saveas(gcf, ['result/ARX_Model_Validation_na_value_', num2str(na_uji(i)), '.png']);
    title('ARX Model Validation for nb value -', num2str(nb_uji(i)));
    saveas(gcf, ['result/ARX_Model_Validation_nb_value_', num2str(nb_uji(i)), '.png']);

    subplot(2,1,2);
    plot(test_data.OutputData, 'b'); hold on;
    plot(y_test_pred.OutputData, 'r--'); hold off;
    legend('Actual (Test)', 'Predicted (Test)');
    xlabel('Time (hours)');
    ylabel('PV Power');
    % title('ARX Model Testing for na value -', num2str(na_uji(i)));
    % saveas(gcf, ['result/ARX_Model_Testing_na_value_', num2str(na_uji(i)), '.png']);
    title('ARX Model Testing for nb value -', num2str(nb_uji(i)));
    saveas(gcf, ['result/ARX_Model_Testing_nb_value_', num2str(nb_uji(i)), '.png']);
    
    % % Data masa lalu
    % H = 24; % Horizon waktu prediksi (misal 24 jam)
    % test_len = length(test_data.OutputData);
    % n_samples = test_len-(2*H); %mencari datatest yang mempunyai nilai variatif
    % past_data_size = input_day*H;
    % start_data_test = n_samples-H-past_data_size+1;
    % end_data_test = n_samples-H;
    % past_data = test_data(start_data_test:end_data_test); % Ambil 100 sampel pertama sebagai data masa lalu
    % 
    % start_data_fInput = end_data_test+1;
    % end_data_fInput = n_samples;
    % % Data input untuk prediksi masa depan
    % FutureInputs = test_data.InputData(start_data_fInput:end_data_fInput, :); % Ambil 24 sampel input berikutnya
    % figure;
    % forecast(model_arx,past_data,H,FutureInputs);
    % legend('Past Outputs','Future Outputs');
    % title(['ARX Forecasting for Day -', num2str(d)]);
    % saveas(gcf, ['result/ARX_Forecasting_Day_', num2str(d), '.png']);
end
% Save results table as CSV
% results.Properties.VariableNames = {'na value', 'MAE Train', 'RMSE Train', 'FIT Train', 'MAE Test', 'RMSE Test', 'FIT Test'};
results.Properties.VariableNames = {'nb value', 'MAE Train', 'RMSE Train', 'FIT Train', 'MAE Test', 'RMSE Test', 'FIT Test'};
writetable(results, 'result/ARX_Model_Results.csv');
disp('Results saved as ARX_Model_Results.csv');