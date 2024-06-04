function Gamma = GaussianDistanceCovariance(p,sigma,clength)

distances = pdist2(p,p);
Gamma = sigma^2*exp(-distances.^2/(2*clength^2));

end