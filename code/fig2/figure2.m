figure(1)
ezplot('log(1+exp(-x))/log(2)',[-2,2,0,4])
hold on
ezplot('-log(normcdf(x))/log(2)',[-2,2,0,4])
ezplot('x<0',[-2,2,0,4])
xlabel('y_{ij}^u(\theta_i^u+\delta_i^u-\theta_j^u-\delta_j^u+\gamma^u)')
ylabel('Loss')
legend('Bradley-Terry model','Thurstone-Mosteller model','0-1 Loss Function');
title('')
%plot([0,0],[-1,5],'k:')


figure(2)
ezplot('x^2',[-2,2,0,4])
xlabel('y_{ij}^u - (\theta_i^u+\delta_i^u-\theta_j^u-\delta_j^u+\gamma^u)')
ylabel('Loss')
hold on
legend('l_2 loss');
title('')
%plot([0,0],[-1,5],'k:')