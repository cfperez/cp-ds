from sklearn.feature_extraction import DictVectorizer


class DataFrameVectorizer(DictVectorizer):
    '''Scikit-learn transformer for DataFrames. Categorical features become one-hot encoded.
    
    >>> df = pd.DataFrame({'school': ['a','b','c'], 'gpa': [2,3,4]})
    >>> dv = DataFrameVectorizer()
    >>> transformed = dv.fit_transform(df)
    >>> dv.inverse_transform(transformed)
    '''
    def fit(self, x, y=None):
        self._columns = x.columns
        super(DataFrameVectorizer, self).fit(x.to_dict('records'), y)
        self._encoded_columns = dict((feature, feature.split(self.separator))
                      for feature in self.feature_names_
                      if self.separator in feature)
        return self

    def transform(self, x):
        if not set(x.columns) <= set(self._columns):
            raise ValueError('Columns in argument are not a subset of fitted values: %s' 
                             % list(self._columns))
        return super().transform(x.to_dict('records'))
    
    def fit_transform(self, x, y=None):
        X = x.to_dict('records')
        self.fit(x)
        return super(DataFrameVectorizer, self).transform(X)
    
    def _unpivot(self, rowdict):
        for key in rowdict:
            if key in self._encoded_columns:
                del rowdict[key]
                rowdict.update([self._encoded_columns[key]])
        return rowdict

    def inverse_transform(self, x):
        transformed = super(DataFrameVectorizer, self).inverse_transform(x)
        return pd.DataFrame.from_records(
            # Unpivot one-hot encoded columns into single columns
            # This could be faster with a joblib.Parallel implementation
            [self._unpivot(row) for row in transformed],
            columns=self._columns
        # DictVectorizer.inverse_transform() seems to drop 0 values which creates NaNs
        ).fillna(0)
       
    def __repr__(self):
        return 'DataFrameVectorizer({})'.format(getattr(self, 'feature_names_', '<not fit>'))
