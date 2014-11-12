function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 3);
numFilters = size(convolvedFeatures, 4);
convolvedDim = size(convolvedFeatures, 1);

% pooledFeatures = zeros(convolvedDim / poolDim, ...
%         convolvedDim / poolDim, numFilters, numImages);

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 
%   
%   Use mean pooling here.

%%% YOUR CODE HERE %%%
resultDim  = floor(convolvedDim / poolDim);

pooledFeatures = reshape(convolvedFeatures, poolDim, resultDim, poolDim, resultDim, numImages, numFilters);
pooledFeatures = mean(pooledFeatures, 3);
pooledFeatures = mean(pooledFeatures, 1);
pooledFeatures = reshape(pooledFeatures, resultDim, resultDim, numImages, numFilters);

% for imageNum = 1:numImages
%     for featureNum = 1:numFilters
%         for poolRow = 1:resultDim
%             offsetRow = 1+(poolRow-1)*poolDim;
%             for poolCol = 1:resultDim
%                 offsetCol = 1+(poolCol-1)*poolDim;
%                 patch = convolvedFeatures(offsetRow:offsetRow+poolDim-1,offsetCol:offsetCol+poolDim-1,imageNum,featureNum);
%                 pooledFeatures(poolRow,poolCol,featureNum,imageNum) = mean(patch(:));
%             end
%         end
%     end
% end

end

