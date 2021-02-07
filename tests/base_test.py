import unittest
import pipeline_tools as pt
import pandas as pd
import numpy as np


class TestBaseTransformer(unittest.TestCase):
    """
    Base class should leave the data frame unchanged. This class tests this.
    """

    def test_transformation(self):
        test_df = pd.DataFrame(
            {'a': [1, 2, 3], 'b': [5, 6, 6]}
        )
        transformer = pt.BasePipeStep()
        transformed = transformer.fit_transform(test_df).copy()
        self.assertTrue(test_df.equals(transformed))


class SelectColumns(unittest.TestCase):
    """
    Transfromer should filter the columns in the dataframe.
    Data in the columns should be the same
    """

    def test_transformation(self):
        test_df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [5, 6, 6],
            'c': [4, 5, 7]
        })
        columns = ['b', 'c']
        transformer = pt.SelectColumns(columns)
        transformed = transformer.fit_transform(test_df).copy()
        self.assertTrue(test_df[columns].equals(transformed))


class TestHotEncoder(unittest.TestCase):

    test_df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [5, 6, 6],
        'c': ['a', 'b', 'c']
    })
    columns = ['c']
    transformer = pt.OneHotEncoderDf(columns)
    transformer.fit(test_df)

    def test_onehotfit(self):
        """
        fit function should return a copy of the transformer.
        """
        fit_return = self.transformer.fit(self.test_df)
        self.assertTrue(isinstance(fit_return, pt.OneHotEncoderDf))

    def test_onehottransform(self):
        """
        Test a sample df is transformed correctly
        """
        transformed = self.transformer.fit_transform(self.test_df)
        expected = pd.DataFrame({
            'c_b': [0.0, 1.0, 0.0],
            'c_c': [0.0, 0.0, 1.0]
        })
        self.assertTrue(transformed.equals(expected))


class TestScaleNumeric(unittest.TestCase):

    test_df = pd.DataFrame({
        'a': np.random.rand(50),
        'b': np.random.rand(50),
    })
    columns = ['a', 'b']
    transformer = pt.ScaleNumeric(columns)
    transformer.fit(test_df)

    def test_transform(self):
        """
        Check that the scaler is behaving as it should and that only the columns
        we want are getting returned.
        """
        transformed = self.transformer.fit_transform(self.test_df)
        for col in transformed.columns:
            std = np.std(transformed[col])
            mean = np.mean(transformed[col])
            self.assertTrue(std >= 0.97 and std <= 1.03)
            self.assertTrue(mean >= -0.03 and mean <= 0.3)
        self.assertTrue(transformed.columns.tolist() == self.columns)


class TestToNumeric(unittest.TestCase):

    test_df = pd.DataFrame({
        'a': np.random.rand(50),
        'b': np.random.rand(50),
    })
    test_df['a_str'] = test_df.a.astype('str').copy()

    columns = ['a', 'b', 'a_str']
    transformer = pt.ToNumeric(columns)
    transformer.fit(test_df)

    def test_transform(self):
        transformed = self.transformer.fit_transform(self.test_df)
        for col in self.columns:
            self.assertTrue(transformed[col].dtype == 'float64')
        tolerance = 0.000000001
        self.assertTrue(
            transformed[
                (transformed['a_str'] <= (1-tolerance) * transformed['a'])
                |
                (transformed['a_str'] >= (1+tolerance)*transformed['a'])
            ].empty
        )
        
if __name__ == '__main__':
    unittest.main()
