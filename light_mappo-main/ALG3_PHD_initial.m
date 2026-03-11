%{
                           分布式的PHD相关参数进行初始化
%}
for i=1:N_sensor
    
    %***********************
    %   滤波相关参数初始化
    %***********************
    % 全局粒子初始化
    Sensor(i).state{1,1}=0;           % 权重
    Sensor(i).state{2,1}=zeros(6,1);  % 均值
    Sensor(i).state{3,1}=zeros(6,6);  % 协方差
    Sensor(i).state{4,1}=0;           % 粒子数量
    
    %***********************
    %   估计相关参数初始化
    %***********************
    % 第一个时刻的初始值赋值
    Sensor(i).X_est{1,1}=zeros(6,0); % 全局目标估计结果
    Sensor(i).X_est{2,1}=zeros(6,0); % 全局目标估计结果
    
end

% 融合中心变量初始化
Fusion_center = struct;
Fusion_center.sensor_inf =struct;