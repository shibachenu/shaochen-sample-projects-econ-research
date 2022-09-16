x1 = c(8, 4, 7)
x2 = c(2, 8, 1)
x3 = c(3, 1, 1)
x4 = c(9, 7, 4)

X = rbind(x1, x2, x3, x4)

colMeans(X)

n = nrow(X)
ones = matrix(1, n, n)
X_bar = 1/n * t(X) %*% matrix(1, n, 1) 

S = (1/n) * (t(X)%*%X) - X_bar%*%t(X_bar)

S_prime = S*n/(n-1)

cov(X)

X = matrix(c(2, -1, -2))

n = nrow(X)

H = diag(1, n, n) - (1/n)*matrix(1, n, n)

Hx = H %*% X

H2x = H %*% H %*% X

S = (1/n)*t(X)%*%H%*%X

#Projection 
v = matrix(c(1,2))
u = v/norm(v, type = "F")
X1 = c(1,2)
X2 = c(3,4)
X3 = c(-1, 0)

X = cbind(X1, X2, X3)
#projections
Proj_u_scalar = t(u) %*% X 

projX1 = Proj_u_scalar[1][1] * u
projX2 = Proj_u_scalar[2][1] * u
projX3 = Proj_u_scalar[3][1] * u

proj = cbind(projX1, projX2, projX3)

X_signed = t(Proj_u_scalar)

n = nrow(X_signed)

S = (1/n)*t(X_signed) %*% (diag(1, n, n)-(1/n)*matrix(1, n, n)) %*% X_signed

var(X_signed)

(t(u) * S[1][1]) %*% u

#SVD

#correlation matrix (rather than covariance matrix)
M = matrix(rnorm(20),nrow=5)
corM = cor(M)

eigen = eigen(corM)
pca <- prcomp(M, center = TRUE,scale. = TRUE)

summary(pca)

pca$rotation
pca$x

#install.packages("devtools")
#library(devtools)
#install_github("vqv/ggbiplot")
plot(pca$x[,1], pca$x[,2])


#Conceptual Example I: 2 data points in 2D

x1 = c(0,1)
x2 = c(0,-1)

X = rbind(x1, x2)
n = nrow(X)
S = (1/(n))*t(X)%*%X

eigen(S)

cov(X)

#Conceptual Examples in 2 dimensions
x1 = c(1,1/2)
x2 = c(-1, -1/2)

plot(x1, x2)

X = rbind(x1, x2)

S = (1/nrow(X))*t(X)%*%X

eigen(S)

prcomp(X)

PC1 = matrix(c(-1, -1/2))
PC1_unit = PC1/norm(PC1, type = "F")

#signed distance
t(PC1_unit) %*% X

#11. (Optional) Conceptual Examples in 2 dimensions

x1 = c(0,2)
x2 = c(1,-1)
x3 = c(-1,-1)

X = rbind(x1, x2, x3)

plot(X)

S = (1/nrow(X))*(t(X)%*%X)

eigen = eigen(S)

PC1 = matrix(eigen$vectors[,1])
PC2 = matrix(eigen$vectors[,2])

y = X %*% PC1
var(y)

#12. Conceptual Examples Continued
X = rbind(c(0,2), c(0,-2), c(1,1), c(-1,-1))

S = (1/nrow(X))*(t(X)%*%X)

eigen(S)
