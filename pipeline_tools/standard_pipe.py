"""
Generating the standard pipeline that will get us 90% to a lovely model.
"""
from sklearn.pipeline import Pipeline
from .simple_transformers import SelectColumns, OneHotEncoderDf, ToNumeric, ScaleNumeric, FillNumericData
from .feature_union_df import FeatureUnionDf

def standard_preprocessing_pipe(num_features=[], cat_features=[]):

    cat_prepipe = Pipeline([
        ('select_cols', SelectColumns(cat_features)),
        ('one_hot',OneHotEncoderDf(cat_features))
    ])

    numeric_prepipe = Pipeline([
        ('select_cols', SelectColumns(num_features)),
        ('cast_as_float', ToNumeric(num_features)),
        ('impute', FillNumericData(num_features)),
        ('scale_feautes', ScaleNumeric(num_features)),
    ])

    preprocessing = FeatureUnionDf([
        ('numeric_pipe', numeric_prepipe),
        ('cat_prepipe', cat_prepipe)
    ])

    return preprocessing


