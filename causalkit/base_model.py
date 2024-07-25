from enum import Enum

class ModelType(Enum):
    RandomForestClassifier = 1
    RandomForestRegressor = 2

class PyModel:
    """
    this is an abstract model class that all causal model in python should inherited from
    """
    def __init__(self, model_type, params):
        """
            model_type (ModelType): 
                the causalmodel to be used as listed in the Enum ModelType
            params (dict):
                - feature (List[str]): list of input features; should be the same order for train/test
                - cat (List[str] = []): features in this list will be treated as categorical features
                - treatment (List[str] = []): list of treatment columns, support only one treatment column for now
                - y (str = ""): the response column
                - weight (str = ""): weight column, if empty, all samples in the dataset have equal weights
                - seed (int = None): random seed
                - **: any other hyperparameters that specific to your model
        """
        self._model_type = model_type
        self.params = params

    @property
    def model_type(self):
        return self._model_type
    
    def fit(self, columns, array):
        """
            columns (List[str]): the column names for array; if a column name is in `columns` 
                but not in self.params["feature"], it will not be used as input feature for training
            array (np.array): 2-dim numpy array, each row is a record
        """
        raise NotImplementedError

    def predict(self, columns, array):
        """
            columns (List[str]): the column names for array; if a column name is in `columns` 
                but not in self.params["feature"], it will not be used as input feature for predict
            array (np.array): 2-dim numpy array, each row is a record

        return
            score (np.array): 2-dim NxT numpy matrix, N is #record, T is #treatment. for example, 
            if there are two groups control/treatment, then T=1. the score is Nx1 matrix, with value
            the uplift of Prob(Y|treatment) - Prob(Y|control)
        """
        raise NotImplementedError
    
    def load(self, path):
        """
            path (str): disk location of the model file
            return
                model (PyModel): the model with all model parameters loaded

        """
        raise NotImplementedError
    
    def save(self, path):
        """
            path (str): save to disk
        """
        raise NotImplementedError
