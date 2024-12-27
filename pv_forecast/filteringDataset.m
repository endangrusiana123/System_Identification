% Memuat data
data = readtable('data/CleanedDataset.csv');

% Kolom yang akan diperbaiki
pv_power = data.PV_Power_Mean;
g_h = data.G_h_Mean;
g_tilt = data.G_tilt_Mean;
w_s = data.W_s_Mean;
w_d = data.W_d_Mean;

% Menggunakan Savitzky-Golay Filter untuk smoothing
frameSize = 11; % Panjang jendela filter
polyOrder = 2;  % Orde polinomial

pv_power_filtered = sgolayfilt(pv_power, polyOrder, frameSize);
g_h_filtered = sgolayfilt(g_h, polyOrder, frameSize);
g_tilt_filtered = sgolayfilt(g_tilt, polyOrder, frameSize);

% Menggunakan Median Filter untuk data dengan spike/outlier
w_s_filtered = medfilt1(w_s, 5); % Jendela ukuran 5
w_d_filtered = medfilt1(w_d, 5); % Jendela ukuran 5

% Menambahkan kolom yang telah diperbaiki ke tabel
data.PV_Power_Filtered = pv_power_filtered;
data.G_h_Filtered = g_h_filtered;
data.G_tilt_Filtered = g_tilt_filtered;
data.W_s_Filtered = w_s_filtered;
data.W_d_Filtered = w_d_filtered;

% Mengambil data jam dari kolom HourlyTime
datetimeValues = datetime(data.HourlyTime, 'InputFormat', 'dd-MMM-yyyy HH:mm:ss');
hours = hour(datetimeValues);

% Menambahkan kolom baru dengan jam sebagai integer
data.HourInteger = hours;

% Mengambil data jam dari kolom HourlyTime
datetimeValues = datetime(data.HourlyTime, 'InputFormat', 'dd-MMM-yyyy HH:mm:ss');
hours = hour(datetimeValues);

% Menyimpan data yang telah diperbaiki ke file baru
writetable(data, 'data/CleanedDataset_Filtered.csv');

% Menampilkan pesan berhasil
disp('Data telah diperbaiki dan disimpan dalam file CleanedDataset_Filtered.csv');
