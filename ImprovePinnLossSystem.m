%% train_pinn_msd.m — PINN
clear; clc; close all; rng(0);

% ===== 1) Load data =====
load massSpringDamperData.mat
% נדרש: tdata(1xN), xdata(1xN), tpinns(1xM), m,mu,k,t0,tmax
assert(numel(tdata)==numel(xdata), 'tdata and xdata must have same length');

% ===== 2) Normalize time to tau in [0,1] + dlarray('CB') =====
tScale       = (tmax - t0);
tau_data     = reshape((tdata  - t0) / tScale, 1, []);
tau_pinns    = reshape((tpinns - t0) / tScale, 1, []);
xdata        = reshape(xdata, 1, []);

tau_data_dl  = dlarray(tau_data,  'CB');   % (C=1, B=N)
xdata_dl     = dlarray(xdata,     'CB');
tau_pinns_dl = dlarray(tau_pinns, 'CB');

% רשת להערכה / ציור
teval      = linspace(t0, tmax, 400);
taueval_dl = dlarray((teval - t0)/tScale, 'CB');

% ===== 3) Build tiny network t->x(t) =====
inputSize=1; outputSize=1; hiddenUnits=32;     % קטן ויציב
layers = [
    featureInputLayer(inputSize, Name="in")
    fullyConnectedLayer(hiddenUnits, Name="fc1")
    tanhLayer(Name="tanh1")
    fullyConnectedLayer(hiddenUnits, Name="fc2")
    tanhLayer(Name="tanh2")
    fullyConnectedLayer(outputSize, Name="out")
];
net = dlnetwork(layers);

% ===== 4) Auto-balance weights for losses (חשוב לפתיחה יציבה) =====
Ldata0 = dlfeval(@(n,td,xd) dataLossOnly(n,td,xd), net, tau_data_dl, xdata_dl);
Lphys0 = dlfeval(@(n,tp)     pinnsLossOnly(n,tp,m,mu,k,tScale), net, tau_pinns_dl);
W_data  = 1.0;
W_pinns = 0.000000001 * double(extractdata( Ldata0 / (Lphys0 + eps) ));   % פיזיקה מתחילה חלש יחסית
fprintf('Auto weights: W_data=%.3g, W_pinns=%.3g\n', W_data, W_pinns);

% ===== 5) Training hyperparameters =====
numIters  = 2000;
learnRate = 1e-3;        % עדין יותר ל-PINNs
np = []; vp = [];        % Adam states
lossHist = zeros(numIters,3);  % [Ldata, Lphys, Ltot]

% ===== 6) Training loop =====
for it = 1:numIters
    [L, gNet, parts] = dlfeval(@modeLoss, net, ...
        tau_data_dl, xdata_dl, tau_pinns_dl, m, mu, k, tScale, W_data, W_pinns);

    % עדכון עם Adam
    [net, np, vp] = adamupdate(net, gNet, np, vp, it, learnRate);

    % לוג
    lossHist(it,:) = gather([parts.data, parts.phys, L]);
    if mod(it,100)==0 || it==1
        gnorm = gradGlobalNorm(gNet);    % נורמת גרדיאנט
        fprintf('Iter %4d | Ldata=%.3e  Lphys=%.3e  Ltot=%.3e  | grad-norm=%.2e\n', ...
            it, lossHist(it,1), lossHist(it,2), lossHist(it,3), gnorm);

        % תצוגה חיה
        x_pred_eval = predict(net, taueval_dl);
        figure(10); clf; hold on; grid on;
        plot(tdata, xdata, 'o', 'DisplayName','noisy xdata');
        if exist('xclean','var')
            plot(tdata, xclean, '-', 'LineWidth',1.5, 'DisplayName','clean (analytic)');
        end
        plot(teval, extractdata(x_pred_eval), '--', 'LineWidth',2, 'DisplayName','PINN (data+physics)');
        xlabel('t'); ylabel('x(t)'); legend('Location','best');
        title(sprintf('Data+Physics @ iter %d', it));
        drawnow;
    end
end

% ===== 7) Final plots =====
x_pred_eval = predict(net, taueval_dl);

figure('Name','PINN vs data/clean'); hold on; grid on;
plot(tdata, xdata, 'o', 'DisplayName','noisy xdata');
if exist('xclean','var'), plot(tdata, xclean, '-', 'LineWidth',1.5, 'DisplayName','clean'); end
plot(teval, extractdata(x_pred_eval), '--', 'LineWidth',2, 'DisplayName','PINN (data+physics)');
xlabel('t'); ylabel('x(t)'); legend('Location','best'); title('Final');

figure('Name','Loss curves'); hold on; grid on;
plot(lossHist(:,1),'LineWidth',1.5); 
plot(lossHist(:,2),'LineWidth',1.5);
plot(lossHist(:,3),'k','LineWidth',2);
xlabel('iteration'); ylabel('loss'); legend('data','physics','total');

%% ================= Loss functions =================
function [Loss, grad, parts] = modeLoss(net, tau_data, xdata, tau_pinns, m, mu, k, tScale, W_DATA, W_PHYS)
    % --- Data loss ---
    Ldata = dataLossOnly(net, tau_data, xdata);

    % --- Physics loss ---
    Lphys = pinnsLossOnly(net, tau_pinns, m, mu, k, tScale);

    % --- Total loss + gradients w.r.t. Learnables ---
    Loss = W_DATA*Ldata + W_PHYS*Lphys;
    grad = dlgradient(Loss, net.Learnables);

    if nargout>=3
        parts.data = gather(Ldata);
        parts.phys = gather(Lphys);
    end
end

function L = dataLossOnly(net, tau_data, xdata)
    xhat = forward(net, tau_data);         % 1xN (CB)
    L    = l2loss(xhat, xdata);            % scalar MSE
end

function L = pinnsLossOnly(net, tau_pinns, m, mu, k, tScale)
    % x(τ)
    x   = forward(net, tau_pinns);                                     % 1xM (CB)
    % נגזרות לפי τ (הקפדה על סקלר לפני dlgradient)
    xt  = dlgradient(sum(x,"all"),  tau_pinns, EnableHigherDerivatives=true);
    xtt = dlgradient(sum(xt,"all"), tau_pinns);
    % מעבר לנגזרות בזמן t
    v = xt  / tScale;               % dx/dt
    a = xtt / (tScale^2);           % d2x/dt2
    % שארית ה-ODE
    r = m.*a + mu.*v + k.*x;
    % לוס פיזיקה כסקלר
    L = mean(r.^2, "all");
end

function gnorm = gradGlobalNorm(gNet)
    % נורמת גרדיאנט גלובלית לצורכי דיאגנוסטיקה
    s = 0;
    for i = 1:size(gNet,1)
        gi = gNet.Value{i};
        s = s + sum(gi.^2,'all');
    end
    gnorm = sqrt(gather(s));
end
