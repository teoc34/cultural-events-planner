import numpy as np  # used for numerical operation on array
import sklearn.preprocessing as pp  # used for standardizing data

# np.ndarray(shape=(rows, columns)
x = np.ndarray()  # standardized
y = np.ndarray()  # standardized

# number of observations
# p, q number of variables
n, p = x.shape
q = y.shape[1]
Cov = np.cov(x, y, rowvar=False)  # calculates the covariance matrix, treating columns as variables
CX = np.cov(x, rowvar=False)  # computes the covariance matrices for x and y
CY = np.cov(y, rowvar=False)
invCX = np.linalg.inv(CX)  # inverts the cov matrices using linear algebra
invCY = np.linalg.inv(CY)
CXY = Cov[:p, p:]  # extracts the cross-covariance x and y from the combined matrix
CYX = CXY.T  # transposes for CYX
h1 = invCX @ CXY  # calculates the canonical coefficients by multiplying the inverse cov with cross-cov
h2 = invCY @ CYX
# eigen decomposition
# code computes the eigenvalues and eigenvectors for the matrices
# fundamental in CCA to find linear correlations that maximize the correlation between datasets
m = min(p, q)
if p == m:
    h = h1 @ h2
    r2, a = np.linalg.eig(h)
    r = np.sqrt (r2)
    b = (h2 @ a) @ np.diag(1 / r)
else:
    h = h2 @ h1
    r2, b = np.linalg.eig(h)
    r = np.sqrt (r2)
    a = (h1 @ b) @ np.diag(1 / r)

# normalize the projections of x and y on their canonical variables(a, b) across columns
z = pp.normalize(x @ a, axis=0)
u = pp.normalize(y @ b, axis=0)

# Calculates the correlation matrices between the original variables and their canonical variables
# for both x and y, using only the relevant dimensions (m).
Rxz = np.corrcoef(x, z[:, :m], rowvar=False)[:p, p:]
Ryu = np.corrcoef(y, u[:, :m], rowvar=False)[:q, p:]



