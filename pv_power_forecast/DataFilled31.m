% Baca dataset
data = readtable('data/HourlyDataset-SolarTechLab-Filtered.csv');

% Konversi HourlyTime ke tipe datetime
data.HourlyTime = datetime(data.HourlyTime, 'InputFormat', 'dd-MMM-yyyy HH:mm:ss');

% Buat variabel untuk tanggal tanpa jam
data.DateOnly = dateshift(data.HourlyTime, 'start', 'day');

% Tentukan rentang tanggal (diasumsikan data penuh untuk satu tahun)
startDate = datetime(2017, 1, 1);
endDate = datetime(2017, 12, 31);
allDates = startDate:endDate;

% Pisahkan per hari dan cek data hilang
uniqueDates = unique(data.DateOnly);
missingDates = setdiff(allDates, uniqueDates);

% Data baru dengan hari-hari yang diisi
newData = data;

% Tambahkan data untuk hari-hari yang hilang
for i = 1:length(missingDates)
    missingDate = missingDates(i);

    % Cari hari terdekat yang tersedia
    previousDate = uniqueDates(find(uniqueDates < missingDate, 1, 'last'));
    nextDate = uniqueDates(find(uniqueDates > missingDate, 1, 'first'));
    
    if isempty(previousDate)
        nearestDate = nextDate;
    elseif isempty(nextDate)
        nearestDate = previousDate;
    else
        % Pilih hari terdekat
        if abs(missingDate - previousDate) <= abs(nextDate - missingDate)
            nearestDate = previousDate;
        else
            nearestDate = nextDate;
        end
    end
    
    % Salin data 24 jam dari hari terdekat
    nearestData = data(data.DateOnly == nearestDate, :);
    copiedData = nearestData;
    copiedData.DateOnly = repmat(missingDate, size(nearestData, 1), 1);
    copiedData.HourlyTime = copiedData.HourlyTime + (missingDate - nearestDate);
    
    % Tambahkan ke dataset baru
    newData = [newData; copiedData];
end

% Urutkan data berdasarkan waktu
newData = sortrows(newData, 'HourlyTime');

% Simpan ke file baru
writetable(newData, 'data/HourlyDataset-Filled.csv');
