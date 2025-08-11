% MSIVY algorithm standalone version
% Application of multi-strategy integrated Ivy algorithm in acoustic metasurface design

clear; clc; close all; warning off all;

%% 【1】 Global parameter definition
max_iter  = 100;    % Maximum number of iterations
earlyStop = 40;     % Threshold for early stopping of consecutive no-progress iterations
freq      = 3e3;   % sound wave frequency (Hz)
v         = 1500;  % Speed of sound (m/s)
lambda0   = v/freq; % wavelength(m)
k0        = 2*pi/lambda0; % Wave number
d         = lambda0;      % Cell spacing
a         = 9;      % Hypersurface side length (number of elements)
Npop      = 10;     % Population size
dim       = a * a;  % Solution vector dimension
levels    = (0:7) * (2*pi/8); % Phase discretization level (0 to 2pi, 8 levels)
lb        = 1;      %Index lower bound
ub        = numel(levels); % Index upper bound

%% 【2】 initialization
% Randomly generate a set of phase indices and then copy Npop copies to make all initial individuals consistent
idx0    = randi([lb, ub], 1, dim);         % Randomly generate a 1×dim index vector
init_idx = repmat(idx0, Npop, 1);          % Copy to Npop×dim
init_pos = levels(init_idx);               % Get the corresponding phase vector

%% 【3】Running the MSIVY algorithm
[msivy_bestSol, msivy_convCurve, msivy_iter] = ...
    MSIVY(Npop, max_iter, earlyStop, k0, d, levels, lb, ub, init_pos);
msivy_bestPhase = reshape(msivy_bestSol.Position, a, a);

%% 【4】 Output
fprintf('MSIVY最终指标: 均值=%.6f, 方差=%.6f, 最大值=%.6f (迭代 %d)\n', ...
        msivy_bestSol.Mean, msivy_bestSol.Var, msivy_bestSol.Max, msivy_iter);

%% 【5】 Draw the convergence curve
figure; hold on;
plot(0:msivy_iter-1, msivy_convCurve(1:msivy_iter,1), 'r-s', 'LineWidth',1.5);
plot(0:msivy_iter-1, msivy_convCurve(1:msivy_iter,2), 'g-^', 'LineWidth',1.5);
plot(0:msivy_iter-1, msivy_convCurve(1:msivy_iter,3), 'b-o', 'LineWidth',1.5);

xlabel('Number of iterations'); 
ylabel('Objective function value');
legend('Peak mean','peak variance','peak maximum','Location','best'); 
grid on;
title('Convergence curve of MSIVY algorithm');

%% ===== Core implementation of MSIVY algorithm =====
function [bestSol, convCurve, actual_iter] = MSIVY(Npop, MaxIter, earlyStop, k0, d, levels, lb, ub, init_pos)
    % Initialize dimensions and parameters
    dim = size(init_pos, 2);
    delta = levels(2) - levels(1);  

    % Initialize the population structure
    pop = repmat(struct('Position',[], 'Mean',[], 'Var',[], 'Max',[], 'Cost',[]), Npop, 1);
    for i = 1:Npop
        pop(i).Position = init_pos(i,:);
        [pop(i).Mean, pop(i).Var, pop(i).Max] = uniformity_metrics(init_pos(i,:), k0, d);
        pop(i).Cost = pop(i).Mean + pop(i).Var + pop(i).Max;
    end

    % Initialize the optimal solution
    [~, ibest] = min([pop.Cost]);
    bestSol = pop(ibest);
    convCurve = zeros(MaxIter+1, 3);
    convCurve(1,:) = [bestSol.Mean, bestSol.Var, bestSol.Max];
    lastImp = 1;
    actual_iter = MaxIter;

    % Main iteration process
    for it = 1:MaxIter
        beta = 2 - 1.8*(it / MaxIter)^3;  % Strategy C: Nonlinear Decay Factor
        newpop = repmat(pop(1), Npop, 1);

        for i = 1:Npop
            nb = mod(i, Npop) + 1;  
            if pop(i).Cost < beta * bestSol.Cost
                D_base = abs(randn(1,dim)) .* (pop(nb).Position - pop(i).Position) + randn(1,dim) .* pop(i).Position/(ub-lb);
            else
                D_base = bestSol.Position .* (rand(1,dim) + randn(1,dim) .* pop(i).Position/(ub-lb)) - pop(i).Position;
            end

            % Strategy A: Lévy perturbation
            levy_step = 0.2 * levy_flight(dim, 1.5) * exp(-5 * it / MaxIter);
            if pop(i).Cost < beta * bestSol.Cost
                D = D_base + levy_step;
            else
                D = bestSol.Position .* (rand(1,dim) + 0.8 * randn(1,dim) .* pop(i).Position/(ub-lb)) - pop(i).Position;
            end
            cand = pop(i).Position + D;

            % Strategy D: Three candidate solutions
            idx_q = round((cand - levels(1)) / delta) + 1;
            idx_q = min(max(idx_q, 1), numel(levels));
            qpos = levels(idx_q);
            [cost_q_mean, cost_q_var, cost_q_max] = uniformity_metrics(qpos, k0, d);
            cost_q = cost_q_mean + cost_q_var + cost_q_max;

            idx_opp = numel(levels) + 1 - idx_q;
            opp = levels(idx_opp);
            [cost_opp_mean, cost_opp_var, cost_opp_max] = uniformity_metrics(opp, k0, d);
            cost_opp = cost_opp_mean + cost_opp_var + cost_opp_max;

            sigma = (MaxIter - it) / MaxIter * delta * 2;
            randp = quantize_phase(pop(i).Position + randn(1,dim) * sigma, levels, lb, ub);
            [cost_r_mean, cost_r_var, cost_r_max] = uniformity_metrics(randp, k0, d);
            cost_rand = cost_r_mean + cost_r_var + cost_r_max;

            costs = [cost_q, cost_opp, cost_rand];
            [~, bc] = min(costs);
            if bc == 1
                newpop(i).Position = qpos;
                newpop(i).Mean = cost_q_mean;
                newpop(i).Var = cost_q_var;
                newpop(i).Max = cost_q_max;
            elseif bc == 2
                newpop(i).Position = opp;
                newpop(i).Mean = cost_opp_mean;
                newpop(i).Var = cost_opp_var;
                newpop(i).Max = cost_opp_max;
            else
                newpop(i).Position = randp;
                newpop(i).Mean = cost_r_mean;
                newpop(i).Var = cost_r_var;
                newpop(i).Max = cost_r_max;
            end
            newpop(i).Cost = costs(bc);
        end

        % Merge populations and select the best
        pop = [pop; newpop];
        [~, ord] = sort([pop.Cost]);
        pop = pop(ord(1:Npop));

        % Strategy B: Local Differential Evolution (LDE)
        F = 0.7; nLDE = 3;
        for e = 1:2
            for k = 1:nLDE
                idxs = randperm(Npop, 3);
                xr1 = pop(idxs(1)).Position;
                xr2 = pop(idxs(2)).Position;
                xr3 = pop(idxs(3)).Position;
                v = xr1 + F * (xr2 - xr3);
                vq = quantize_phase(v, levels, lb, ub);
                [costv_mean, costv_var, costv_max] = uniformity_metrics(vq, k0, d);
                costv = costv_mean + costv_var + costv_max;
                if costv < pop(e).Cost
                    pop(e).Position = vq;
                    pop(e).Mean = costv_mean;
                    pop(e).Var = costv_var;
                    pop(e).Max = costv_max;
                    pop(e).Cost = costv;
                end
            end
        end

        % Update optimal solution and convergence curve
        if pop(1).Cost < bestSol.Cost
            bestSol = pop(1);
            lastImp = it;
        elseif it - lastImp >= earlyStop
            actual_iter = it;
            convCurve = convCurve(1:actual_iter+1, :);
            break;
        end
        convCurve(it+1,:) = [bestSol.Mean, bestSol.Var, bestSol.Max];
    end
end

%% ===== Dependent helper functions=====
function q = quantize_phase(phi, levels, lb, ub)
    idx = round((phi - levels(1)) / ((levels(end)-levels(1))/(ub-lb))) + 1;
    idx = min(max(idx, lb), ub);
    q   = levels(idx);
end

function [mean_val, var_val, max_val, peaks, coords] = uniformity_metrics(phi_vec, k0, d)
    a = sqrt(numel(phi_vec));
    phi_mat = reshape(phi_vec, a, a);
    [TH, PH] = meshgrid(linspace(0, pi/2, 200), linspace(0, 2*pi, 200));
    E  = calculate_E_total(phi_mat, k0, d, TH, PH);
    Ez = abs(E) .* cos(TH);
    [peaks, coords] = find_local_peaks(Ez, TH, PH);
    
    % Limit peak quantity
    if numel(peaks) > 40
        [~, I] = sort(peaks, 'descend');
        peaks  = peaks(I(1:40));
        coords = coords(I(1:40), :);
    end
    
    mean_val = mean(peaks);  % Mean
    var_val = var(peaks);    % Variance
    max_val = max(peaks);    %Max Peak
end

function [peaks, coords] = find_local_peaks(Ez, TH, PH)
    scales = [3,5,7]; allp=[]; allc=[];
    for s = scales
        se = strel('disk', s);
        dil = imdilate(Ez, se);
        [r, c] = find(Ez == dil);
        for i = 1:numel(r)
            allp(end+1)   = Ez(r(i), c(i));
            allc(end+1,:) = [PH(r(i), c(i)), TH(r(i), c(i))];
        end
    end
    [~, o] = sort(allp, 'descend'); keep = true(size(o));
    for i = 1:length(o)-1
        for j = i+1:length(o)
            if abs(allp(o(i)) - allp(o(j))) < 1e-3, keep(j)=false; end
        end
    end
    peaks = allp(o(keep));
    coords= allc(o(keep), :);
end

function E = calculate_E_total(phi_mat, k0, d, TH, PH)
    a = size(phi_mat,1); E = zeros(size(TH));
    for m = 1:a
        for n = 1:a
            phase = (m-0.5)*k0*d .* sin(TH).*cos(PH) + ...
                    (n-0.5)*k0*d .* sin(TH).*sin(PH) + phi_mat(m,n);
            E = E + exp(-1i * phase);
        end
    end
end

function L = levy_flight(d, beta)
    sigma_u = (gamma(1+beta)*sin(pi*beta/2) / ...
        (gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
    nu = randn(1,d) * sigma_u;
    w  = randn(1,d);
    L  = 0.01 * (nu ./ abs(w).^(1/beta));
end