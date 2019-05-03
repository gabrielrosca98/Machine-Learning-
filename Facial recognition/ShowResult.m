function []=ShowResult(data, labels, prediction, N)
%
% Visualises the classification mistakes of the supplied face examples against their true labels.
%
% SHOWDATA(data, labels, prediction, N)
%
% Arguments: 
%   'data' should be a matrix of n examples (n rows) and d dimensions (d columns).
%   'labels' contains the true labels of the supplied examples.
%   'predictions' contains the predicted labels of the supplied examples.
%   'N' should be an integer specifying how many examples are shown per row.
%    
% Returns: Nothing.
% 
% Both 'labels' and 'predictions' should be vectors of the same length n.
% 'data' MUST be the original pixels (i.e. 1024 columns in the data matrix).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  YOU SHOULD NOT BE EDITING THIS CODE
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
   faceW = 32; 
   faceH = 32; 
   
   
   n = size(data,1);
   
   if N>n
      N=n;
   end
   

 
   
   numPerLine = N; 
   if mod(n,numPerLine)~= 0
       ShowLine = floor(n/numPerLine)+1;
   else
       ShowLine = n/numPerLine;
   end
   border=3;
   
   Y = zeros(faceH*ShowLine,faceW*numPerLine); 
   for i=0:ShowLine-1 
      for j=0:numPerLine-1 
         if i*numPerLine+j+1<=n
            if labels(i*numPerLine+j+1) == prediction(i*numPerLine+j+1)
               Y(i*faceH+1:(i+1)*faceH,j*faceW+1:(j+1)*faceW) = reshape(data(i*numPerLine+j+1,:),[faceH,faceW]); 
            else
               frame = zeros(32);
               frame(2:30,2:30) =255;
               content = reshape(data(i*numPerLine+j+1,:),[faceH,faceW]); 
               frame(border+1 : 31-border, border+1 : 31-border) = content(border+1 : 31-border, border+1 : 31-border);
               Y(i*faceH+1:(i+1)*faceH,j*faceW+1:(j+1)*faceW) = frame; 
            end
         else 
            break
         end
      end 
   end 
   
  
   imagesc(Y);colormap(gray);
   
end