import unittest
import pipeline_tools as pt
import pandas as pd
import numpy as np

class GenericTransformerTest(unittest.TestCase):
    test_df = pd.DataFrame(
        {'a': [1, 2, 3], 'b': [5, 6, 6]}
    )
    TransformerClass = pt.BasePipeStep
    transformer = TransformerClass()

    def test_fitReturn(self):
        """
        fit function should return a copy of the transformer.
        """
        fit_return = self.transformer.fit(self.test_df)
        self.assertTrue(isinstance(fit_return, self.TransformerClass))



class TestBaseTransformer(GenericTransformerTest):
    """
    Base class should leave the data frame unchanged. This class tests this.
    """

    def test_transformation(self):
        transformed = self.transformer.fit_transform(self.test_df).copy()
        self.assertTrue(self.test_df.equals(transformed))


class SelectColumns(GenericTransformerTest):
    """
    Transfromer should filter the columns in the dataframe.
    Data in the columns should be the same
    """
    test_df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [5, 6, 6],
            'c': [4, 5, 7]
        })
    columns = ['b', 'c']
    TransformerClass = pt.SelectColumns
    transformer = TransformerClass(columns)

    def test_transformation(self):
        transformed = self.transformer.fit_transform(self.test_df).copy()
        self.assertTrue(self.test_df[self.columns].equals(transformed))


class TestHotEncoder(GenericTransformerTest):

    test_df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [5, 6, 6],
        'c': ['a', 'b', 'c']
    })
    columns = ['c']
    TransformerClass = pt.OneHotEncoderDf
    transformer = TransformerClass(columns)

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


class TestScaleNumeric(GenericTransformerTest):

    test_df = pd.DataFrame({
        'a': np.random.rand(50),
        'b': np.random.rand(50),
    })
    columns = ['a', 'b']
    TransformerClass = pt.ScaleNumeric
    transformer = TransformerClass(columns)


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


class TestToNumeric(GenericTransformerTest):

    test_df = pd.DataFrame({
        'a': np.random.rand(50),
        'b': np.random.rand(50),
    })
    test_df['a_str'] = test_df.a.astype('str').copy()

    columns = ['a', 'b', 'a_str']
    TransformerClass = pt.ToNumeric
    transformer = TransformerClass(columns)

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

class TestStandardPipe(unittest.TestCase):
    test_df = pd.DataFrame({
        'a': np.random.rand(50),
        'b': np.random.rand(50),
        'c': [str(int(x*4)) for x in np.random.rand(50)],
    })
    num_features = ['a', 'b']
    cat_features = ['c']

    def test_pipe(self):
        preprocessing = pt.standard_preprocessing_pipe(
            num_features=self.num_features,
            cat_features=self.cat_features
        )
        transformed = preprocessing.fit_transform(self.test_df)
        print(transformed)



        
if __name__ == '__main__':
    unittest.main()
