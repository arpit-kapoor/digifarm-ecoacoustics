import numpy as np


from sklearn.metrics import (mean_squared_error, 
                             mean_absolute_error, 
                             mean_absolute_percentage_error)

class CrossValidation(object):

    _model_types = ['reg', 'ts']
    _metric_dict = {
        'mse': mean_squared_error,
        'mae': mean_absolute_error,
        'mape': mean_absolute_percentage_error
    }

    def __init__(self, model, k=4, model_type='reg', metric='mse') -> None:
        
        self.model = model
        
        assert model_type in self._model_types, 'model_type must be one of {}'.format(self._model_types)
        
        self.model_type = model_type
        self.metric_func = self.initialise_metric(metric)
        self.k = k

    def _split_data(self, X, y=None, k=4):
        """split data for k-fold cross-validation"""
        if self.model_type=='reg':
            n = len(X)
            indices = np.arange(n)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            return np.array_split(X, k), np.array_split(y, k)
        elif self.model_type=='ts':
            return np.array_split(X, k), None
        
    def initialise_metric(self, metric:str):
        metric_func = self._metric_dict.get(metric)
        if metric_func is None:
            raise ValueError('metric must be one of {}'.format(self._metric_dict.keys()))
        return metric_func

    def run(self, X, y=None):
        """run cross-validation"""
        self._split_X, self._split_y = self._split_data(X, y, self.k)

        self._scores = []

        if self.model_type == 'reg':
            for i in range(self.k):

                # Train set
                X_train = np.concatenate(self._split_X[:i] + self._split_X[i+1:], axis=0)
                y_train = np.concatenate(self._split_y[:i] + self._split_y[i+1:], axis=0)
                
                # Train and validate
                self.model.train(X_train, y_train)
                self._predictions = self.model.predict(self._split_X[i])
                
                if isinstance(self._predictions, tuple):
                    self._predictions = self._predictions[0]

                # Evaluate
                score = self.metric_func(self._split_y[i], self._predictions)

                # Append score
                self._scores.append(score)
        
        elif self.model_type == 'ts':

            for i in range(1, self.k):

                # Train set
                X_train = np.concatenate(self._split_X[:i], axis=0)

                # Train and validate
                self.model.train(X_train)
                self._predictions = self.model.predict(steps=len(self._split_X[i]))

                if isinstance(self._predictions, tuple):
                    self._predictions = self._predictions[0]

                # Evaluate
                score = self.metric_func(self._split_X[i], self._predictions)

                # Append score
                self._scores.append(score)

        # Calculate mean score
        self.mean_score = np.array(self._scores).mean()

        return self.mean_score


        