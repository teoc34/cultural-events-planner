import numpy as np
import sklearn.cross_decomposition as skl

x = np.ndarray()
y = np.ndarray()

n, p = x.shape  # shapes of arrays are used to extract the no of variables
q = y.shape[1]
m = min(p, q)  # m calculates the min to determine the no of comp used in CCA
# ensure that no of comp does not exceed the dimension of the smaller set

# CCA model object is instantiated from scikit-learn's cross_decomposition
# CCA class, specifying n_components as m(number of canonical components to compute)
# The fit method is then called with x and y to fit the CCA model to the datasets.
modelCCA = skl.CCA(n_components=m)
modelCCA.fit(x, y)
# transform method applies the learned CCA mapping to x and y,
# producing the canonical variables z and u that represent x and y in the canonical correlation space
# where the correlations between z and u are maximized.
z, u = modelCCA.transform(x, y)

# These lines retrieve the loadings (canonical coefficients)
# for x and y from the fitted CCA model.
# x_loadings_ and y_loadings_ represent the correlation between the original variables and
# the canonical variables for datasets x and y, respectively.
Rxz = modelCCA.x_loadings_
Ryu = modelCCA.y_loadings_