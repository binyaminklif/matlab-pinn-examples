g = 9.81;
m = 1;
v_0 = 3;
y_0 = 2;
k = 0.1;
e = 0.6;
r = 0.1;
T = 10;
dt = 0.001;
N = T/dt + 1;
V = zeros(1,N);
Y = zeros(1,N);
V(1) = v_0;
Y(1) = y_0;
n = 1;

for t = 0:dt:T
    a_drag = -k*V(n)/m ;
    V(n + 1) = V(n) - (g - a_drag)*dt;
    Y(n + 1) = Y(n) + V(n)*dt - 0.5*(g - a_drag)*dt*dt;
    if Y(n + 1)<= r
        Y(n + 1) = r;
        V(n + 1) = -V(n)*e;
    end
    n = n+1;
end
t = 0:dt:(T+dt)
[a,b] = findPeaksManual(t,Y)
plot(t,Y)
xlabel time(s)
ylabel y(m)
hold on
plot(t,V)

function [peaks, locs] = findPeaksManual(x, y)
   
    if length(x) ~= length(y)
        error('x and y must have the same length');
    end

    peaks = [];
    locs = [];

    for i = 2:length(y)-1
        if y(i) > y(i-1) && y(i) > y(i+1)
            peaks(end+1) = y(i);      % מוסיפים את ערך הפיק
            locs(end+1) = x(i);       % מוסיפים את המיקום המתאים
        end
    end
end

