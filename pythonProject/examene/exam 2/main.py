import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

rawIndicators = pd.read_csv('./dataIN/GlobalIndicatorsPerCapita_2021.csv', index_col=0)
rawContinents = pd.read_csv('./dataIN/CountryContinents.csv', index_col=0)
labInd = list(rawIndicators.columns.values[1:])
ind = list(rawIndicators.index.values)

merged=rawIndicators.merge(raw.Continents, left_index=True, right_index=True)\.drop('Country_y')