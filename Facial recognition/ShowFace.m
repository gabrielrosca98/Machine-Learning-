function []=ShowFace(data, N)

% Visualizes the supplied face data matrix.
%
% ShowFace(data, N)
%
% Arguments: 
%   'data' should be a matrix of n examples (n rows) and d dimensions (d columns).
%   'N' should be an integer specifying how many examples are shown per row.
% 
%Returns: Nothing.
% 
% MUST be the original pixels (i.e. 1024 columns in the data matrix)
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
   if mod(n,numPerLine) ~= 0
       ShowLine = floor(n/numPerLine)+1; 
   else
       ShowLine = n/numPerLine;
   end

   Y = zeros(faceH*ShowLine,faceW*numPerLine); 
   for i=0:ShowLine-1 
      for j=0:numPerLine-1 
         if i*numPerLine+j+1<=n
            Y(i*faceH+1:(i+1)*faceH,j*faceW+1:(j+1)*faceW) = reshape(data(i*numPerLine+j+1,:),[faceH,faceW]); 
         else 
            break
         end
      end 
   end 

  
   imagesc(Y);colormap(gray);
   
end