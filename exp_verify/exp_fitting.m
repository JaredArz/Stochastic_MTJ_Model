% Data pulled from Thermal effects in spin torque switching of perpendicular magnetic tunnel junctions at cryogenic temperatures, Rehm et. al
Temps = [4 35 75 110 150 185 220 255 295]';
Temps_fine = 0:1:300; % to be used with analytical curves

% Magnetic saturation in A/m
Ms = [580769.230769;
     573076.923077;
     553846.153846;
     542307.692308;
     526923.076923;
     514615.384615;
     504807.692308;
     487980.769231;
     473076.923077];
% normalization factor, smallest measured value at 4 Kelvin
Mstar = Ms(1);

% Effective anisotropy energy in J/m^2 calculated from Hk_eff measurements
K = [0.0005756874999997713;
    0.0005736500000000771;
    0.0005467679999998481;
    0.0005305125000003011;
    0.0005137499999999251;
    0.0004961772299996292;
    0.0004648415625002834;
    0.00044538200000021063;
    0.00041205000000006707];

% To use power law, convert to J/m^3 and solve for K1 with magnetostatic anisotropy term
K = ((K ./ (2.6*10^(-9))) + ((Ms.^2)/2)*(4*pi*10^(-7))); % power law is J/m^3

MoverMstar = Ms/Mstar;
MoverMstar_fine = linspace(MoverMstar(1),MoverMstar(length(MoverMstar)) , 301);


% fit Ms exp data to blochs law
bloch_func = fittype( @(Tc,alpha,x) Mstar*( 1 - (x./Tc).^alpha ) );
[bloch_law_fitted_curve, gof] = fit(Temps,Ms,bloch_func,'StartPoint',[500 1])
Ms_coeffs = coeffvalues(bloch_law_fitted_curve);
Tc = Ms_coeffs(1);
alpha = Ms_coeffs(2);

figure;
scatter(Temps, Ms, ...
        'b+', ...
        'LineWidth',10)
hold on
plot(Temps, bloch_law_fitted_curve(Temps), ...
    'r', ...
    'LineWidth',2)


set(gca,'fontsize', 18) 
hold off
xlabel('T')
ylabel ('Ms (A/m)')
legend('exp. data','Blochs law')



% fit calcuated K data to power law
power_func = fittype( @(n,Kstar,x) Kstar * ( 1 - (x/Tc).^alpha ).^n );
[power_curve,gof] = fit(Temps, K, power_func, 'StartPoint', [1 1000])
K_coeffs = coeffvalues(power_curve);
n = K_coeffs(1);
Kstar = K_coeffs(2);
KoverKstar = K/Kstar;

figure;

scatter(Temps, K, ...
    'b+', ...
    'LineWidth',10)
hold on
plot(Temps, power_curve(Temps),...
    'r', ...
    'LineWidth',2)

set(gca,'fontsize', 18) 
hold off
xlabel('T (K)')
ylabel ('K (J/m^3)')
legend('exp. data','Power law fit')




% not necessary, but fit again using normalized values
%norm_power_func = fittype( @(x) x.^n );
%[norm_power_curve,gof] = fit(MoverMstar,KoverKstar,norm_power_func,'StartPoint', 1)


figure;
loglog(MoverMstar, KoverKstar, ...
    'b+', ...
    'LineWidth',10)
hold on
loglog(MoverMstar,MoverMstar.^n, ...
    'r', ...
    'LineWidth',2)
set ( gca, 'xdir', 'reverse');
set(gca,'fontsize', 18) 
hold off
xlabel('M/M(4K)')
ylabel ('K/K(4K)')
legend('exp. data','Power law fit')



% tmr = [2.20 2.17 2.11 2.03 1.98 1.24];
% T_tmr = [4 35 75 110 150 295]';
% P = (tmr./(2+ tmr)).^0.5;
% P = P';
% p_func = fittype( @(P0,Tcc,x) P0 * ( 1 - (x/Tcc).^() ));
% [p_curve,gof] = fit(T_tmr, P, p_func, 'StartPoint', [0.5, 1000])
% 
% figure;
% 
% scatter(T_tmr, P, ...
%     'b+', ...
%     'LineWidth',10)
% hold on
% plot(T_tmr, p_curve(T_tmr),...
%     'r', ...
%     'LineWidth',2)
% 
% set(gca,'fontsize', 18) 
% hold off
% xlabel('T (K)')
% ylabel ('P')
% legend('exp. data','fit')
% 
% 

