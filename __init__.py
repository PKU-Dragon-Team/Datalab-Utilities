import json
import numpy as np
import pandas as pd
import numbers


class NumpyAndPandasEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, np.matrix)):
            return [self.default(x) for x in obj]
        if isinstance(obj, pd.DataFrame):
            return [self.default(series) for _, series in obj.iterrows()]
        if isinstance(obj, pd.Series):
            return [self.default(val) for _, val in obj.iteritems()]
        if isinstance(obj, numbers.Integral):
            return int(obj)
        if isinstance(obj, (numbers.Real, numbers.Rational)):
            return float(obj)
        return super().default(obj)
