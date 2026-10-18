% clear; clc;

n_points = 100; 
samples_per_model = 1000;

models = {'ogden1','ogden2','ogden3',...
          'red_poly1','red_poly2','red_poly3',...
          'poly1','poly2','poly3'};

data_csv = [];
model_names = {};

rng(42); %Random Seed for Reproducibility

for m = 1:length(models)
    model_name = models{m};

    for sample = 1:samples_per_model
        strain = sort(rand(n_points,1) * 3); %generate random strain values (0-3), sorted in acending order
        lambda = 1 + strain;

        params = nan(1,9); %NaN placeholder for params

        switch model_name
            case 'ogden1'
                scale = 0.1;
                %randn generates a random number from a standard normal distribution (mean = 0, std = 1),  
                params(1) = (5.0 + randn*0.5) * scale; %mu1
                params(2) = 0.5 + randn*0.05; %alpha1
                stress = (2*params(1)/params(2)) .* (lambda.^params(2) - lambda.^(-params(2)/2));

            case 'ogden2'
                scale = 0.01;
                params(1) = (6.0 + randn*0.6) * scale; %mu1
                params(2) = 0.01 + randn*0.001; %alpha1
                params(3) = (2.0 + randn*0.2) * scale; %mu2
                params(4) = -5.0 + randn*0.5; %alpha2
                stress = (2*params(1)/params(2)) .* (lambda.^params(2) - lambda.^(-params(2)/2)) + ...
                         (2*params(3)/params(4)) .* (lambda.^params(4) - lambda.^(-params(4)/2));

            case 'ogden3'
                scale = 0.05;
                params(1) = (6.0 + randn*0.6) * scale; %mu1
                params(2) = 0.01 + randn*0.001; %alpha1
                params(3) = (2.0 + randn*0.2) * scale; %mu2
                params(4) = -5.0 + randn*0.5; %alpha2
                params(5) = (0.1 + randn*0.01) * scale; %mu3
                params(6) = 0.1 + randn*0.01; %alpha3
                stress = (2*params(1)/params(2)) .* (lambda.^params(2) - lambda.^(-params(2)/2)) + ...
                         (2*params(3)/params(4)) .* (lambda.^params(4) - lambda.^(-params(4)/2)) + ...
                         (2*params(5)/params(6)) .* (lambda.^params(6) - lambda.^(-params(6)/2));

            case 'red_poly1'
                scale = 0.1;
                params(1) = (1.0 + randn*0.1) * scale; %C10
                stress = 2*params(1) .* (lambda.^2 - lambda.^(-1));

            case 'red_poly2'
                scale = 0.01;
                params(1) = (5 + randn*0.5) * scale; %C10
                params(2) = (0.2 + randn*0.02) * scale; %C20
                stress = 2*params(1) .* (lambda.^2 - lambda.^(-1)) + ...
                         4*params(2) .* (lambda.^2 - lambda.^(-1)) .* (lambda.^2 + 2*lambda.^(-1) - 3);

            case 'red_poly3'
                scale = 0.04;
                params(1) = (5 + randn*0.5) * scale; %C10
                params(2) = (-0.3 + randn*0.001) * scale; %C20
                params(3) = (0.01 + randn*0.0001) * scale; %C30
                A = lambda.^2 + 2*lambda.^(-1) - 3;
                stress = 2*(lambda.^2 - lambda.^(-1)) .* ...
                         (params(1) + 2*params(2).*A + 3*params(3).*A.^2);

            case 'poly1'
                scale = 0.01;
                params = nan(1,9); 
                params(1) = (3 + randn*0.3) * scale;   %C10
                params(2) = (10 + randn*1) * scale;    %C01
                stress = 2*(lambda.^2 - lambda.^(-1)) .* ...
                         (params(1) + lambda.^(-1).*params(2));
    
            case 'poly2'
                scale = 0.01;
                params = nan(1,9);
                params(1) = (3 + randn*0.3) * scale;       %C10
                params(2) = (10 + randn*1) * scale;        %C01
                params(3) = (0.12 + randn*0.012) * scale;  %C20 
                params(5) = (0.02 + randn*0.002) * scale;  %C02
                params(4) = (0.06 + randn*0.006) * scale;  %C11
                I1 = lambda.^2 + 2*lambda.^(-1);
                I2 = lambda.^(-2) + 2*lambda;
                A = I1 - 3; B = I2 - 3;
                stress = 2*(lambda.^2 - lambda.^(-1)) .* ...
                         (params(1) + 2*params(3).*A + params(4).*B) ...
                         - 4*(lambda.^(-2) - lambda) .* ...
                         (params(2) + params(4).*A + 2*params(5).*B);

            case 'poly3'
            scale = 0.01;
            params = nan(1,9);
            params(1) = (3 + randn*0.3) * scale;         %C10
            params(2) = (10 + randn*1) * scale;          %C01 
            params(3) = (-0.12 + randn*0.012) * scale;   %C20
            params(5) = (0.02 + randn*0.002) * scale;    %C02
            params(6) = (0.015 + randn*0.0015) * scale;  %C30 
            params(9) = (0.004 + randn*0.0004) * scale;  %C03 
            params(4) = (-0.06 + randn*0.006) * scale;   %C11
            params(8) = (0.01 + randn*0.001) * scale;    %C12
            params(7) = (0.004 + randn*0.0004) * scale;  %C21
            I1 = lambda.^2 + 2*lambda.^(-1);
            I2 = lambda.^(-2) + 2*lambda;
            A = I1 - 3; B = I2 - 3;
            stress = 2*(lambda.^2 - lambda.^(-1)) .* ...
                     (params(1) + 2*params(3).*A + params(4).*B + ...
                      3*params(6).*(A.^2) + 2*params(7).*A.*B + params(8).*(B.^2)) ...
                     - 4*(lambda.^(-2) - lambda) .* ...
                     (params(2) + params(4).*A + 2*params(5).*B + ...
                      params(7).*(A.^2) + 2*params(8).*A.*B + 3*params(9).*(B.^2));
        end

        %add random noice (3%-5%) to half of the samples
        if rand < 0.5
            noise_level = 0.03 + (0.05 - 0.03) * rand; % 3% to 5%
            stress = stress .* (1 + noise_level * randn(size(stress)));
        end
    
        %store model names, stress,strain, parameters (all the columns)
        model_names{end+1,1} = model_name;
        row_data = [stress' strain' params];
        data_csv = [data_csv; row_data];
    end
end

%Build table with stores columns
stress_cols = arrayfun(@(i) sprintf('stress_%d',i), 1:n_points, 'UniformOutput', false);
strain_cols = arrayfun(@(i) sprintf('strain_%d',i), 1:n_points, 'UniformOutput', false);
param_cols  = arrayfun(@(i) sprintf('param_%d',i), 1:9, 'UniformOutput', false);
T = array2table(data_csv, 'VariableNames', [stress_cols strain_cols param_cols]);
T = addvars(T, model_names, 'Before', 1, 'NewVariableNames', 'model');

%Save to CSV
writetable(T, 'training_dataset.csv');
disp('CSV saved: training_dataset.csv');

%=========================
%Plot 100 Samples per Model
%=========================
T = readtable('training_dataset.csv');
models = unique(T.model, 'stable');

figure;
tiledlayout(3,3, 'Padding', 'compact', 'TileSpacing', 'compact');

for m = 1:numel(models)
    nexttile;
    hold on;

    %Get only this model's rows
    T_model = T(strcmp(T.model, models{m}), :);

    %Randomly pick 100 samples from this model
    sample_idx = randperm(height(T_model), min(100, height(T_model)));

    for i = sample_idx
        stress_vals = table2array(T_model(i, 2:101)); 
        strain_vals = table2array(T_model(i, 102:201));
        plot(strain_vals, stress_vals, 'LineWidth', 1);
    end

    xlabel('Strain');
    ylabel('Stress (MPa)');
    title(strrep(models{m}, '_', ' '), 'FontWeight', 'bold');
    grid on;
end

sgtitle('Synthetic Stress-Strain Curves (100 Samples per Model)');