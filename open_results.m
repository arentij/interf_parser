file_name_par = 'CMFX_01095_scope_results_fft.parquet';
data1 =parquetread(file_name_par);
%%

lambda_red = 632e-9;
lambda_co2 = 10.56e-6;

distance_co2 = lambda_co2/2/pi * unwrap(data1.phase_differences_co2_fft);
distance_red = lambda_red/2/pi * unwrap(data1.phase_differences_red_fft);
time_arr = data1.t_fft;
%%
figure(1)
clf
hold on
plot(time_arr,-distance_co2 + mean(distance_co2(1:1000)))
plot(time_arr, distance_red - mean(distance_red(1:1000)))