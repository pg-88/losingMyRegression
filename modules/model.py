import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import mlem
import numpy as np

class Model():
    def __init__(self, data_csv) -> None:
        '''
        Create the data frame and the variables to use for the model


        Args:
            data_csv: file or link that contain the csv file
        '''
        
        try:
            self.df = pd.read_csv(data_csv)
        except:
            print(f"Error {data_csv} is not readable as csv file.")

        self.X = pd.DataFrame() # target
        self.y = pd.DataFrame() # feature

    def get_col_names(self):
        '''return list with the names of columns'''
        #print(self.df)
        cols = self.df.columns
        return cols
    
    def _set_feature(self, col_name):
        '''modify the variable feature (X) to isolate the feature'''
        if col_name in self.df.columns:
            self.X = self.df.drop(columns=col_name) 

        else:
            print(f"Error {col_name} is not a name for a column in this dataframe.")



    def _set_target(self, col_name):
        '''modify the variable target (y) to separate from the feature'''
        if col_name in self.df.columns:
            self.y = self.df[col_name]
        else:
            print(f"Error {col_name} is not a name for a column in this dataframe.")

    def _get_train_test(self, targ_col, test_set=0.33) -> tuple:
        '''after calling the set method for X and y, split them in 2 set

        use 42 for the random state parameter.

        Args:
            targ_col: name of the colunm selected as target
            test_set: float (between 0 and 1) that represent 
                the size percentage of the test set (default 0.33 or 1/3)
        return:
            tuple of pandas series: feature train, feature test,
                                    target train, target test.'''
        self._set_feature(targ_col)
        self._set_target(targ_col)
        
        return train_test_split(self.X, self.y, test_size=test_set, random_state=42)


class SingleReg(Model):

    def __init__(self, data_csv) -> None:
        super().__init__(data_csv)
        self.model = LinearRegression()
        self.err = []

    def _set_feature(self, col_name):
        '''modify the variable feature (X) to isolate the feature'''
        if col_name in self.df.columns:
            self.X = self.df.drop(columns=col_name) 
            # after drop the last column is the one with data
            # linear single regression X has only one column of data
            col = self.X.columns[-1]
            self.X = self.X[col]
            self.X = self.X.to_numpy().reshape((-1,1))

        else:
            print(f"Error {col_name} is not a name for a column in this dataframe.")

    def _gen_model(self, target):
        X_train, X_test, y_train, y_test = self._get_train_test(target)

        #X_train = X_train.transpose()
        #X_test = X_test.transpose()
        print("transposed feature", X_train.shape)                
        self.model.fit(X=X_train, y=y_train)
        ## DEBUG this
        self._calc_err(X_test, y_test)
    
    def _calc_err(self, x, y) -> dict:
        '''calculate errors of the model'''
        y_pred = self.model.predict(x)
        if y_pred.size == y.size:
            print(y_pred - y)
        else: print(f"oh-oh.. predicted has size {y_pred.size}, while test set has size {y.size}")
        return {
            "R_square": round(r2_score(y_true=y, y_pred=y_pred),4),
            "MeanAbsErr": round(mean_absolute_error(y_true=y, y_pred=y_pred),4),
            "RootMeanSquareErr": round(mean_squared_error(y_true=y, y_pred=y_pred, squared=False),4)
            }

    def save_model(self, target):
        '''generate bin for model with mlem 
        
        Args:
            target: name of the target column
        Return:
            errors of model as dict {R_square: float, MeanAbsErr: float, RootMeanSquareErr: float}'''
        self._gen_model(target)
        #mlem.api.save(self.model, "resources/my_model", "resources/my_model_", self.df)
        return self.err





class MultipleReg(Model):
    pass 


class Classification(Model):
    pass

def test():
    '''generate a linear variable to test linear model'''
    samp_size = 500
    m = 0.88 * np.random.rand(samp_size) # inclination of line with random rumors
    i = 30 * np.random.rand(samp_size) # intercept 

    x = 10 * np.random.randint(3, 1000, samp_size) # value on x axis of observations
    y = m * x + i # value on y axis of observations (dependent variable)

    p = pd.DataFrame(data=y, columns=["y"])
    p.insert(0,"x", x)
    #print(p)
    p.to_csv("test1.csv")

    test = SingleReg("test1.csv")
    print(test.save_model("y"))
    

if __name__ == '__main__':
    test()
