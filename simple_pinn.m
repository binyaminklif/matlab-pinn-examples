%% 1. Build neural network
inputSize = 1;
outputSize = 1;
numHiddenUnits = 20;

layers = [
    featureInputLayer(1, "Normalization", "none")
    fullyConnectedLayer(numHiddenUnits)
    tanhLayer
    fullyConnectedLayer(outputSize)
];

net = dlnetwork(layers);

%% 2. Training loop
numIterations = 3000;
learnRate = 0.001;

t_train = linspace(0, 1, 50)';  % Collocation points

gradDecay = 0.9;
sqGradDecay = 0.999;
trailingAvg = [];
trailingAvgSq = [];

for epoch = 1:numIterations
    % === חשוב! המרה ל־dlarray כאן ===
    t_dl = dlarray(t_train', "CB");
    t_dl = dlarray(t_dl, "CB");
   

    
    % חישוב פונקציית עלות וגזירתה
    [loss, grad] = dlfeval(@modelLoss, net, t_dl);
    
    % עדכון פרמטרים עם ADAM
    [net, trailingAvg, trailingAvgSq] = adamupdate(net, grad, ...
        trailingAvg, trailingAvgSq, epoch, learnRate, gradDecay, sqGradDecay);
    
    % הדפסת איבוד כל 100 איטרציות
    if mod(epoch, 100) == 0
        disp("Epoch " + epoch + ", Loss = " + extractdata(loss));
    end
end

%% 3. Prediction
t_test = linspace(0, 1, 100)';
t_test_dl = dlarray(t_test', "CB");
y_pred = predict(net, t_test_dl);

% Plot
plot(t_test, extractdata(y_pred), 'r', 'LineWidth', 2);
hold on
plot(t_test, exp(-t_test), 'b--'); % Exact solution
legend("Neural Network", "Exact");
xlabel("t"), ylabel("y(t)")
title("PINN Solution for dy/dt = -y")


%% 4. Loss function
function [loss, gradient] = modelLoss(net, t_dl)
    % חישוב הפלט y(t)
    y = forward(net, t_dl);

    % גזירה לפי t בעזרת autodiff
    dy = dlgradient(sum(y), t_dl);

    % חישוב פונקציית ההפרה f = dy + y
    f = dy + y;
    loss_f = mean(f.^2);

    % תנאי התחלה y(0) = 1
    y0 = forward(net, dlarray(0, "CB"));
    loss_ic = (y0 - 1)^2;

    % פונקציית עלות כוללת
    loss = loss_f + loss_ic;

    % נגזרת של פונקציית העלות לפי פרמטרי הרשת
    gradient = dlgradient(loss, net.Learnables);
end
