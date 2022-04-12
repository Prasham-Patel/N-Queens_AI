function pm = point_mass(Kp,Kd,Ki,m,F,time)

t = 0:0.001:time;
x_0 = 0;
x = x_0;
xd = 1;

i=2;
e_prev(1) = 1;
v(1) = 0;
E = 0;
e(1) = 1;

while i<=length(t)
    e(i) = xd - x(i-1);
    e_prev(i) = e(i-1);
    E = E+e(i);
    de(i) = abs(e(i) - e_prev(i))/0.001;
    F_e(i) = Kp*e(i) + Ki*E + Kd*de(i);
    F_t(i) = F + F_e(i);
    a(i) = F_t(i)/m;
    v(i) = v(i-1) + 0.001*a(i);
    x(i) = v(i)*0.001 + 0.5*a(i)*0.001*0.001;
    i = i+1;
end

% i=1;
% figure
% while i<=length(t)
%     plot(t(:),1)
%     plot(x(i),1,'-o','LineWidth',2)
%     xlim([-0.2,1.2])
%     i = i+1;
%     pause(0.0001)
% end

plot(t,x(:))
pks = findpeaks(abs(x-xd));
summ = sum(pks,'all');

fprintf('Final Error:\n')
e(end)

fprintf('Total Overshoots:\n')
length(pks)

fprintf('Total Overshoot Amplitude:\n')
summ

pm = [e(end); length(pks); summ];

end