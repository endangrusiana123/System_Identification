% Close all figures
close all;

% Load data from CSV
data = readtable('data/CleanedDataset_Filtered.csv');

% Extract important columns
hourInt = data.HourInteger;
pv_power = data.PV_Power_Filtered;  % PV Power Output
temperature = data.T_air_Mean;      % Input: Air Temperature
irradiance = data.G_h_Filtered;     % Input: Global Horizontal Irradiance
tilt = data.G_tilt_Filtered;        % Input: Global Tilt Irradiance
wind_speed = data.W_s_Filtered;     % Input: Wind Speed
wind_direction = data.W_d_Filtered; % Input: Wind Direction

hourInt(1:12)
pv_power(1:12)
temperature(1:12)
tilt(1:12)
wind_speed(1:12)
wind_direction(1:12)

% Combine inputs into a matrix
inputs = [hourInt, pv_power, irradiance, temperature, tilt, wind_speed, wind_direction];

% Shift PV power forward by 24 rows
shift = 24; % Number of rows to shift
output = [pv_power(shift+1:end); NaN(shift, 1)]; % Add NaN for shifted rows

% Truncate inputs to match output length
inputs = inputs(1:end-shift, :); % Remove last 'shift' rows from inputs
% time = time(1:end-shift);        % Adjust time to match inputs length

% Remove NaN rows from output
validIdx = ~isnan(output);
inputs = inputs(validIdx, :);
output = output(validIdx);

disp(['inputs size: ', num2str(size(inputs))])
disp(['output size: ', num2str(size(output))])

results = table();
seqOption = [1, 4, 8 , 24];
for countSeq = 1:length(seqOption)
    % Parameters
    seqLen = seqOption(countSeq); % Number of time steps in each sequence
    disp(["seqLen:",num2str(seqLen)]);
    % Initialize variables
    X = {}; % Sequential inputs
    Y = []; % Responses (now a numeric array for simplicity)
    % Create sequential data
    n = size(inputs, 1); % Total number of data points
    
    for i = 1:(n - seqLen + 1)
        % Create input sequence of size 24x5
        X{end+1} = inputs(i:i+seqLen-1, :); % Keep in [24x5] format
        Y(end+1) = output(i+seqLen-1);         % Only the last time step
    end
    % Convert to a cell array for compatibility
    X = X';
    Y = Y';
    
    disp(['X size: ', num2str(size(X))])
    X(1:4)
    
    disp(['Y size: ', num2str(size(Y))])
    Y(1:4)
    
    muX = mean(cell2mat(X));
    sigmaX = std(cell2mat(X),0);
    
    disp(['Class of Y: ', class(Y)]);
    disp(['Size of Y: ', num2str(size(Y))]);
    disp('First few elements of Y:');
    disp(Y(1:min(3, end))); % Display first 3 elements of Y
    
    % Calculate mean and standard deviation
    muY = mean(Y);    % Mean of Y
    sigmaY = std(Y);  % Standard deviation of Y
    %Y
    % Display results
    disp(['Variance of Y: ', num2str(var(Y))]);
    disp(['Mean of Y: ', num2str(muY)]);
    disp(['Standard deviation of Y: ', num2str(sigmaY)]);
    
    for n = 1:numel(X)
        X{n} = (X{n} - muX) ./ sigmaX;
    end
    % Normalize Y (numeric array)
    Y = (Y - muY) ./ sigmaY; % Normalize Y directly
    
    % Verify normalization of X
    allFeaturesNormalized = vertcat(X{:});
    disp('Mean of normalized features (X):');
    disp(mean(allFeaturesNormalized)); % Should be close to zero
    disp('Standard deviation of normalized features (X):');
    disp(std(allFeaturesNormalized)); % Should be close to one
    
    % Verify normalization of Y
    disp('Mean of normalized Y:');
    disp(mean(Y)); % Should be close to zero
    disp('Standard deviation of normalized Y:');
    disp(std(Y)); % Should be close to one
    
    % Calculate indices for splitting
    numSamples = length(X); % Total number of samples
    
    trainIdx = floor(0.4 * numSamples); % 50% for training
    valIdx = trainIdx + floor(0.2 * numSamples); % 10% for validation (after training)
    
    % Split data into training, validation, and test sets
    X_train = X(1:trainIdx); % First 50% for training
    Y_train = Y(1:trainIdx)'; % Corresponding Y values for training
    
    X_val = X(trainIdx+1:valIdx); % Next 10% for validation
    Y_val = Y(trainIdx+1:valIdx)'; % Corresponding Y values for validation
    
    X_test = X(valIdx+1:end); % Remaining 40% for testing
    Y_test = Y(valIdx+1:end)'; % Corresponding Y values for testing
    
    % Display sizes of each split
    disp(['Number of training samples: ', num2str(length(X_train))]);
    disp(['Number of validation samples: ', num2str(length(X_val))]);
    disp(['Number of test samples: ', num2str(length(X_test))]);
    
    % Ensure Y_train and Y_test are column vectors
    Y_train = Y_train(:); % Convert to column vector
    Y_val = Y_val(:);   % Convert to column vector
    Y_test = Y_test(:);   % Convert to column vector
    
    disp(['Number of sequences in X_train: ', num2str(length(X_train))]);
    disp(['Number of responses in Y_train: ', num2str(length(Y_train))]);
    % Check format of X_train
    disp(['X_train size: ', num2str(size(X_train))]);
    disp(['Size of first sequence in X_train: ', num2str(size(X_train{1}))]);
    
    % Check format of Y_train
    disp(['Y_train size: ', num2str(size(Y_train))]);
    disp(['Class of Y_train: ', class(Y_train)]);
    
    
    % Define LSTM network architecture
    inputSize = 7; % Five input variables
    numHiddenUnits = 50; % Number of LSTM units
    outputSize = 1; % One output variable per time step
    
    layers = [
        sequenceInputLayer(inputSize)
        lstmLayer(10, 'OutputMode', 'last')
        fullyConnectedLayer(outputSize)];
    
    options = trainingOptions("adam", ...
        MaxEpochs=50, ...
        SequencePaddingDirection="left", ...
        ValidationData={X_val Y_val}, ...
        Shuffle="every-epoch", ...
        Plots="training-progress", ...
        Verbose=false);
    
    % Train the network
    net = trainnet(X_train, Y_train, layers,"mse", options);
    
    % Evaluate training performance
    % trainPredicted = predict(net, X_train); % Predict for training data
    trainPredicted  = minibatchpredict(net,X_train, ...
        SequencePaddingDirection="left", ...
        UniformOutput=false);
    
    % Convert training predictions and actual responses into numerical arrays
    trainActual = Y_train(:); % Convert to column vector
    % Convert cell array to numeric array
    trainPredictedArray = cell2mat(trainPredicted); % Flatten into a numeric array
    
    disp(['Class of each cell in trainPredicted: ', class(trainPredicted{1})]);
    disp(['Size of the first cell: ', num2str(size(trainPredicted{1}))]);
    
    disp(['Class of Y_train: ', class(Y_train)]);
    disp(['Class of trainPredicted: ', class(trainPredicted)]);
    
    % Calculate MAE for training data
    MAE_train = mean(abs(Y_train - trainPredictedArray));
    
    % Calculate RMSE for training data
    RMSE_train = sqrt(mean((trainActual - trainPredictedArray).^2));
    
    % Calculate Goodness-of-Fit (FIT) for training data
    SStot_train = sum((trainActual - mean(trainActual)).^2); % Total sum of squares
    SSres_train = sum((trainActual - trainPredictedArray).^2); % Residual sum of squares
    FIT_train = (1 - (SSres_train / SStot_train))*100;
    
    % Display training metrics
    fprintf('Training Performance Metrics:\n');
    fprintf('Mean Absolute Error (MAE): %.4f\n', MAE_train);
    fprintf('Root Mean Square Error (RMSE): %.4f\n', RMSE_train);
    fprintf('Goodness-of-Fit (FIT): %.2f%%\n', FIT_train);
    
    % Evaluate test performance
    % testPredicted = predict(net, X_test); % Predict for test data
    testPredicted  = minibatchpredict(net,X_test, ...
        SequencePaddingDirection="left", ...
        UniformOutput=false);
    
    % Convert test predictions and actual responses into numerical arrays
    testActual = Y_test(:); % Convert to column vector
    testPredictedArray =  cell2mat(testPredicted); % Ensure predictions are in column vector form
    
    % Calculate MAE for test data
    MAE_test = mean(abs(testActual - testPredictedArray));
    
    % Calculate RMSE for test data
    RMSE_test = sqrt(mean((testActual - testPredictedArray).^2));
    
    % Calculate Goodness-of-Fit (FIT) for test data
    SStot_test = sum((testActual - mean(testActual)).^2); % Total sum of squares
    SSres_test = sum((testActual - testPredictedArray).^2); % Residual sum of squares
    FIT_test = (1 - (SSres_test / SStot_test))*100;
    
    % Display test metrics
    fprintf('\nTest Data Performance Metrics:\n');
    fprintf('Mean Absolute Error (MAE): %.4f\n', MAE_test);
    fprintf('Root Mean Square Error (RMSE): %.4f\n', RMSE_test);
    fprintf('Goodness-of-Fit (FIT): %.2f%%\n', FIT_test);
    
    results = [results; table(seqOption(countSeq), MAE_train, RMSE_train, FIT_train, MAE_test, RMSE_test, FIT_test)];
    % Plot results
    % figure;
    % plot(testActual, 'b'); hold on;
    % plot(testPredictedArray, 'r--'); hold off;
    % legend('Actual (Test)', 'Predicted (Test)');
    % xlabel('Time (hours)');
    % ylabel('PV Power');
    % title('LSTM Model Testing for PV Power Prediction');
    % Visualize the prediction results
    figure;
    subplot(2,1,1);
    plot(trainActual, 'b'); hold on;
    plot(trainPredictedArray, 'r--'); hold off;
    legend('Actual (Train)', 'Predicted (Train)');
    xlabel('Time (hours)');
    ylabel('PV Power');
    title('LSTM Model Validation for sequence input value -', num2str(seqOption(countSeq)));
    saveas(gcf, ['result/LSTM_Model_Validation_sequence_input_', num2str(seqOption(countSeq)), '.png']);

    subplot(2,1,2);
    plot(testActual, 'b'); hold on;
    plot(testPredictedArray, 'r--'); hold off;
    legend('Actual (Test)', 'Predicted (Test)');
    xlabel('Time (hours)');
    ylabel('PV Power');
    title('LSTM Model Testing for sequence input value -', num2str(seqOption(countSeq)));
    saveas(gcf, ['result/LSTM_Model_Testing_sequence_input_', num2str(seqOption(countSeq)), '.png']);
end
% Save results table as CSV
results.Properties.VariableNames = {'sequence input', 'MAE Train', 'RMSE Train', 'FIT Train', 'MAE Test', 'RMSE Test', 'FIT Test'};
writetable(results, 'result/LSTM_Model_Results.csv');
disp('Results saved as LSTM_Model_Results.csv');