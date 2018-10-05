function feature_vector = Extract_basic_features_cbm (data)
n = size(data, 1);
var_num = size(data, 2);
feature_vector = [];

features = [];
for i = 1:var_num
    seg = data(:,i);
    % 4 features
    features = [features,mean(seg),std(seg),max(seg),min(seg)];
    % 2 features
    features = [features,rssq(seg)/n,rssq(seg-mean(seg))/n];
    % 1 features
    features = [features,mean(abs(seg - mean(seg)))];
    % 10 features
    features = [features,hist(seg, 10)/n];
    % 20 features
    features = [features,hist(seg, 20)/n];
end
    
% 2 features
features = [features,n];
    
%% features vector
feature_vector = [feature_vector; features];

end