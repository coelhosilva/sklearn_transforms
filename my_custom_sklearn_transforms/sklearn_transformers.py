from sklearn.base import BaseEstimator, TransformerMixin


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class ClipColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns, lower_imit, upper_limit):
        self.columns = columns
        self.lower_limit = lower_imit
        self.upper_limit = upper_limit

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        data[self.columns] = data[self.columns].clip(self.lower_limit,self.upper_limit)
        return data
