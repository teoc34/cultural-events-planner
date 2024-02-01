import factor_analyzer as fa
import pandas as pd

x_df = pd.DataFrame() #standardized

kmo = fa.calculate_kmo(x_df)
# kmo[1] must be > 0.6
EFAModel = fa.Factor_analyzer(n_factors=len(x_df.columns.values))
EFAModel.fit(x_df)
factorLoadings = EFAModel.loadings_  # common factors
specificFactors = EFAModel.gen_uniqueness()
eigenValues = EFAModel.get_eigenvalues()  # principal components