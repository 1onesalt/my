%{
    基于有融合中心的算术-几何平均 (AGM) 序贯融合方法
    
    架构参考：ALG3_fusion_GA_distr.m (集中式、序贯融合)
    算法参考：文稿第3.4节 有限视域多传感器融合方法 (状态分布GA融合 + 基数AA融合)
    
    输入：
        sensor_inf:      包含所有传感器局部估计和参数的结构体数组
        match_threshold: 高斯分量关联匹配的阈值 (马氏距离)
    输出：
        Results:         融合后的全局后验强度 (4*1 cell 格式)
%}
function [Results] = Centralized_Fusion_AGM(sensor_inf, match_threshold)

    % === 参数获取 ===
    N_sensor = size(sensor_inf, 2);
    Match_part1_sus = cell(4,1);
    
    % === 序贯融合初始化 ===
    % 找到第一个观测到目标（非空集）的传感器，作为全局融合的初始基准 Match_part1
    ID_record = 0; 
    for i = 1:N_sensor
        if sensor_inf(i).gm_particles{4,1} ~= 0
            Match_part1 = sensor_inf(i).gm_particles;
            ID_record = i;
            break;
        end
    end

    % 判断是否所有传感器都没看到目标，如果都没看到，直接返回空集
    if ID_record == 0
        Results = cell(4,1);
        Results{1,1} = zeros(1,0);  % 权重
        Results{2,1} = zeros(6,0);  % 均值
        Results{3,1} = zeros(6,6*0);% 协方差
        Results{4,1} = 0;           % 数量
        return;
    end
    
    % t_f 用于记录当前参与有效融合的传感器数量，用于动态计算融合权重 \pi_i
    t_f = 1; 
    
    % === 集中式序贯融合主体循环 ===
    for i = ID_record+1 : N_sensor
        
        % 判断当前传感器是否有待匹配分量，若有则提取，若无则跳过
        if sensor_inf(i).gm_particles{4,1} ~= 0
            Match_part2 = sensor_inf(i).gm_particles;
            t_f = t_f + 1; % 参与融合的有效传感器数 +1
        else
            continue;
        end
        
        % 初始化缓存区 (最大可能的分量数为两者之和)
        n_fusion = Match_part1{4,1} + Match_part2{4,1};
        Match_part1_sus{1,1} = zeros(1, n_fusion);
        Match_part1_sus{2,1} = zeros(6, n_fusion);
        Match_part1_sus{3,1} = zeros(6, 6*n_fusion);
        Match_part1_sus{4,1} = 0;
        cnt_AGM = 0; % 实际生成的高斯分量计数
        
        % === 高斯分量匹配 (调用外部匹配程序，基于马氏距离+匈牙利算法) ===
        % 返回 Solution_match(配对索引) 和 Mat_match(0-1关联矩阵)
        [Solution_match, Mat_match] = ALG3_Match(Match_part1, Match_part2, match_threshold);
        
        % 动态计算 AGM 的融合权重 \pi_1 和 \pi_2
        % 序贯融合中，积累的 part1 权重占比为 (t_f-1)/t_f，当前 part2 占比为 1/t_f
        pi_1 = 1 - 1/t_f;
        pi_2 = 1/t_f;
        
        % =========================================================
        % 1. 匹配组融合 (AGM 核心公式)
        % =========================================================
        for j = 1:size(Solution_match, 1)
            if Solution_match(j) ~= 0
                k = Solution_match(j); % j 对应 part1 分量, k 对应 part2 分量           
                cnt_AGM = cnt_AGM + 1; 
                
                % (1) 协方差的 GA 融合 (公式 3.24)
                % P_AGM = ( pi_1 * P_1^(-1) + pi_2 * P_2^(-1) )^(-1)
                P1_inv = inv(Match_part1{3,1}(:, 6*(j-1)+1 : 6*j));
                P2_inv = inv(Match_part2{3,1}(:, 6*(k-1)+1 : 6*k));
                Match_part1_sus{3,1}(:, 6*(cnt_AGM-1)+1 : 6*cnt_AGM) = inv( pi_1 * P1_inv + pi_2 * P2_inv );
                
                % (2) 状态均值的 GA 融合 (公式 3.25)
                % m_AGM = P_AGM * ( pi_1 * P_1^(-1) * m_1 + pi_2 * P_2^(-1) * m_2 )
                m1 = Match_part1{2,1}(:, j);
                m2 = Match_part2{2,1}(:, k);
                Match_part1_sus{2,1}(:, cnt_AGM) = Match_part1_sus{3,1}(:, 6*(cnt_AGM-1)+1 : 6*cnt_AGM) * ...
                                                   ( pi_1 * P1_inv * m1 + pi_2 * P2_inv * m2 );
                                               
                % (3) 权重的 AA (算术) 融合 (公式 3.23)
                % w_AGM = pi_1 * w_1 + pi_2 * w_2
                w1 = Match_part1{1,1}(1, j);
                w2 = Match_part2{1,1}(1, k);
                Match_part1_sus{1,1}(1, cnt_AGM) = pi_1 * w1 + pi_2 * w2;
            end
        end
        
        % =========================================================
        % 2. 非匹配组融合 (处理有限视域导致的不对称信息)
        % =========================================================
        
        % (1) 遍历 Match_part1 中未被匹配的分量
        for j = 1:size(Mat_match, 1)
            if sum(Mat_match(j, :), 2) == 0 % 第 j 个分量未找到匹配
                cnt_AGM = cnt_AGM + 1; 
                
                Match_part1_sus{3,1}(:, 6*(cnt_AGM-1)+1 : 6*cnt_AGM) = Match_part1{3,1}(:, 6*(j-1)+1 : 6*j);
                Match_part1_sus{2,1}(:, cnt_AGM) = Match_part1{2,1}(:, j);
                
                % 视场判断：判断 part1 的目标是否在当前传感器(part2)的视域内
                flag_FoV = FoV_judge(sensor_inf(i).location, Match_part1{2,1}(:,j), sensor_inf(i).R_detect);
                
                if flag_FoV == 1
                    % 目标在传感器 i 视域内却未被检测到，说明传感器 i 认为其权重为 0
                    % 依据算术融合(AA)法则：w_new = pi_1 * w1 + pi_2 * 0 = pi_1 * w1
                    Match_part1_sus{1,1}(1, cnt_AGM) = pi_1 * Match_part1{1,1}(1, j); 
                else
                    % 目标在传感器 i 视域外，传感器 i 不提供任何有效信息（无权干涉）
                    % 保持其原始权重不变，不进行衰减
                    Match_part1_sus{1,1}(1, cnt_AGM) = Match_part1{1,1}(1, j);
                end
            end
        end
        
        % (2) 遍历 Match_part2 中未被匹配的分量 (新发现的目标)
        for k = 1:size(Mat_match, 2)
            if sum(Mat_match(:, k), 1) == 0 % 第 k 个分量未找到匹配
                cnt_AGM = cnt_AGM + 1; 
                
                Match_part1_sus{3,1}(:, 6*(cnt_AGM-1)+1 : 6*cnt_AGM) = Match_part2{3,1}(:, 6*(k-1)+1 : 6*k);
                Match_part1_sus{2,1}(:, cnt_AGM) = Match_part2{2,1}(:, k);
                
                % 新发现的目标，将其原始权重引入融合池
                Match_part1_sus{1,1}(1, cnt_AGM) = Match_part2{1,1}(1, k);
            end
        end
        
        % =========================================================
        % 3. 数据清理与序贯迭代更新
        % =========================================================
        Match_part1_sus{1,1}(:, cnt_AGM+1:end) = [];
        Match_part1_sus{2,1}(:, cnt_AGM+1:end) = [];
        Match_part1_sus{3,1}(:, 6*cnt_AGM+1:end) = [];
        Match_part1_sus{4,1} = cnt_AGM;
        
        % 将本次融合的结果更新为下一轮序贯融合的基准
        Match_part1 = Match_part1_sus; 
    end
    
    % 输出最终全局融合结果
    Results = Match_part1;

end