% function A = Xraymat_sum_rows(s,phi,N,m)
% 
% The program creates a sparse matrix A used in planar parallel beam
% X-ray tomography.
%
% The image area is the unit square [0,1]x[0,1]. The pixels are enumerated
% in the same order as in matrix indexing, i.e., first pixel is in the top
% left corner, the numbering proceeding columnwise.
%
% Each ray is parametrized by two numbers: phi is the angle between the
% line and vertical axis, phi = 0 corresponding to a ray going upwards,
% phi=-pi/2 going from left to right. The parameter s is the distance
% of the line from the midpoint of the image with sign, so that when phi=0
% and s>0, the line passes through the right half of the image. The line
% can be parametrized as
%
% x(t) = 0.5 + s*cos(phi) - t*sin(phi)
% y(t) = 0.5 + s*sin(phi) + t*cos(phi)
%
% Input: s - a vector containing the s-parameters of the X-rays (of length
%            m times the number of sensors, i.e., rows in A)
%        phi - a vector containing the angles of the X-rays
%        N - integer giving pixel number per edge
%        m - number of rows that are to be summed, i.e., m rays hit each
%            sensor (length of s must be divisible by m)
%
% NOTE: entries of s must lie between -0.5/r and 0.5/r, where 
%       r = max(|sin(phi)|, |cos(\phi)|), otherwise there is no
%       guarantee that they hit the image area and the programme may fail.
%      
% Output: A - sparse matrix of size ((k/m)*j,N*N), k and j being the lengths of the
%             vectors s and phi, respectively.
%
% First version, Erkki Somersalo 10/2/2002
% Version 1.1, Nuutti Hyvonen 12/10/2004
% Version 1.2, Nuutti Hyvonen 12/20018
%-------------------------------------------------------------------------- 
function A = Xraymat_sum_rows(s,phi,N,m)

d1 = length(s);
d2 = length(phi);
s = reshape(s, d1, 1);
phi = reshape(phi, 1, d2);
s = reshape(s*ones(1,d2), d1*d2, 1);
phi = reshape(ones(d1,1)*phi, d1*d2, 1);

%Now [s, phi] contains all distance angle pairs

ss = sin(phi); cc = cos(phi);
K = length(s);
p = 1/N*[0:N]';

AT = [];

Temp = sparse(N*N,m);
for k = 1:K
   txy = [];
   % Finding intersection points with lines y = j/N, 0<=j<=N
   if abs(cc(k)) > eps
      t = 1/cc(k)*(p - 0.5 - s(k)*ss(k));
      x = 0.5 + s(k)*cc(k) -ss(k)*t;
      aux = find((x>=0)&(x<=1));
      txy = [txy;[t(aux),x(aux),p(aux)]];
   end
   
   
   % Finding intersection points with lines x = j/N, 0<=j<=N
   if abs(ss(k)) > eps
      t = 1/ss(k)*(0.5 + s(k)*cc(k) - p);
      y = 0.5 + s(k)*ss(k)  + cc(k)*t;
      aux = find((y>=0)&(y<=1));
      txy = [txy;[t(aux),p(aux),y(aux)]];
   end
   

   % Sorting the intersection points according to increasing t
   
   [tt,I] = sort(txy(:,1));
   txy = [tt,txy(I,2),txy(I,3)];


   % Computing the intersection lengths and pixels.   
   % If the X-ray passes from corner to corner of the pixel, the corner
   % coordinates appear twice. Discarding redundant intersections corresponding
   % to pairs giving zeros (or negligible) intersection length.

   is = length(txy(:,1));
   lengths = txy(2:is,1) - txy(1:is-1,1);
   xmids = 0.5*(txy(2:is,2) + txy(1:is-1,2));
   ymids = 0.5*(txy(2:is,3) + txy(1:is-1,3));
   iaux = find(lengths > eps);
   lengths = lengths(iaux);
   xmids = xmids(iaux);
   ymids = ymids(iaux);
   indx = ceil(N*xmids);
   indy = ceil(N*(1-ymids));

   Temp((indx'-1)*N + indy',mod(k,m-1)+1) = lengths;
   
   % summing the m rows and saving as a column in AT
   if mod(k,m)==0
       kk = k/m;
       AT(:,kk) = sum(Temp,2)/m;
       Temp = sparse(N*N,m);
   end
        
   % monitoring progress
   if(mod(k,5000) == 0)
      display([k,K])
   end
end

A = AT';

return


