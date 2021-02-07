from sklearn.pipeline import FeatureUnion
from scipy import sparse
from sklearn.pipeline import _fit_transform_one, _transform_one
from sklearn.utils._joblib import Parallel, delayed
import pandas as pd
import numpy as np
from functools import reduce

class FeatureUnionDf(FeatureUnion):
    """
    Does the same as sklearns feature union. Apart from if all the transformed outputs are dataframe then the final out put is a 
    dataframe.

    Note that this would be alot simpler if you was using the latest version of sklearn!

    """

    def fit_transform(self, X, y=None, **fit_params):
        """Fit all transformers, transform the data and concatenate results.
        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.
        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.
        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        results = self._parallel_func(X, y, fit_params, _fit_transform_one)
        if not results:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        Xs, transformers = zip(*results)
        self._update_transformer_list(transformers)
        return self._hstack(Xs)

        


    def transform(self, X):
        """Transform X separately by each transformer, concatenate results.
        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.
        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X, None, weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        return self._hstack(Xs)
        
    def _hstack(self, Xs):
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        elif all(isinstance(f, pd.DataFrame) for f in Xs):
            Xs = reduce(lambda df1, df2: df1.join(df2, how='outer'), Xs)
        else:
            Xs = np.hstack(Xs)
        return Xs