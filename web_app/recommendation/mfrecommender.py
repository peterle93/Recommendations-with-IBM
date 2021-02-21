import pandas as pd
import numpy as np
from collections import defaultdict
#import matplotlib.pyplot as plt

class MFRecommender():
    '''
    class: Matrix Factorization
    In this part , matrix factorization is built to make article recommendations to the users on the IBM Watson Studio platform.
    '''
    def __init__(self, articles_clean,interacts_clean,top_n=10):
        self.top_n = top_n
        self.df_content=articles_clean
        self.df = interacts_clean
        self.df_train , self.df_test = self.create_test_and_train_size()

    def create_user_item_matrix(self,df):
        '''
        Description:
        create the user-article matrix with 1's and 0's.
        Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with
        an article and a 0 otherwise
        INPUT:
        self.df - pandas dataframe with article_id, title, user_id columns
        OUTPUT:
        self.user_item - user item matrix
        '''
        # Fill in the function here
        # drop duplicate row and count interact
        user_item=df.drop_duplicates().groupby(['user_id','article_id']).size().unstack(level=1)
        # fill Null to 0
        user_item=user_item.fillna(0)
        # conver float to int
        user_item=user_item.astype('int')
        return user_item # return the user_item matrix

    def create_test_and_train_size(self, train_size=40000, test_size=5993):
         df_train ,df_test= self.df.head(train_size), self.df.tail(test_size)
         return df_train ,df_test

    def create_test_and_train_user_item(self):
        '''
        INPUT:
        df_train - training dataframe
        df_test - test dataframe

        OUTPUT:
        user_item_train - a user-item matrix of the training dataframe
                         (unique users for each row and unique articles for each column)
        user_item_test - a user-item matrix of the testing dataframe
                        (unique users for each row and unique articles for each column)
        test_idx - all of the test user ids
        test_arts - all of the test article ids
        '''
        user_item_train = self.create_user_item_matrix(self.df_train)
        user_item_test = self.create_user_item_matrix(self.df_test)
        test_idx = user_item_test.index.values
        test_arts = user_item_test.columns.values.astype('str')
        return user_item_train, user_item_test, test_idx, test_arts

    def calculate_error(self):
        """
        1 Build the test SVD matrix.
        2 remove test users with cold start issue
        3 draw a curve of  latent factors number vs accuary rate
        """
        # Define the vairable:
        train_errors=[]
        test_errors=[]
        train_accuracy=[]
        test_accuracy=[]
        # create test and train datasets
        user_item_train, user_item_test, test_idx, test_arts = self.create_test_and_train_user_item()
        # fit SVD on the user_item_train matrix
        u_train, s_train, vt_train =np.linalg.svd(user_item_train, full_matrices=False)

        # Build the test SVD matrix.
        ##  Find out user id and article id in both train and test datasets
        pred_users=np.intersect1d(test_idx,user_item_train.index.values)
        pred_arts=np.intersect1d(test_arts,user_item_train.columns.values.astype('str'))
        pred_users_mask=user_item_train.index.isin(pred_users)
        pred_arts_mask= user_item_train.columns.isin(pred_arts)
        ##  Find out U, S,VT matrix based on above user id and article id
        u_test=u_train[pred_users_mask,:]
        s_test=s_train
        vt_test=vt_train[:,pred_arts_mask]

        # remove test users with cold start issue
        user_item_test_actual=user_item_test.loc[pred_users,pred_arts.astype('float')]

        # draw a curve of  latent factors number vs accuary rate
        ## According to latent factors number vs accuary rate curve, find out latent factors number ranges is from 0 to 700.
        ### store the step of latent factor numbe into latent_factors_list.
        latent_factors_num=np.arange(10,700+10,20)
        ## In each step of latent factor number, calculate the errors between actual value and predict value
        for lf in latent_factors_num:
            ### calculate the predict value. np.round will be used becasue interact number belong to [0,1]
            user_item_test_pred=np.round(u_test[:,:lf].dot(np.diag(s_test[:lf]).dot(vt_test[:lf,:])))
            user_item_train_pred=np.round(u_train[:,:lf].dot(np.diag(s_train[:lf]).dot(vt_train[:lf,:])))
            ### calculate errors
            test_diff  = np.subtract(user_item_test_actual,user_item_test_pred)
            train_diff = np.subtract(user_item_train,user_item_train_pred)
            test_error = np.sum(np.sum(np.abs(test_diff)))
            train_error=np.sum(np.sum(np.abs(train_diff)))
            ### store error into a error_list.
            test_errors.append(test_error)
            train_errors.append(train_error)
        test_accuracy= 1 - np.array(test_errors)/(user_item_test_actual.size)
        train_accuracy= 1 - np.array(train_errors)/(user_item_train.size)
        return latent_factors_num,test_accuracy,train_accuracy
    '''
    # Web server use ploly to draw curve, so comment out the following.
    def draw_curve(self,latent_factors_num,test_accuracy,train_accuracy):
        ## Draw the curve
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.plot(latent_factors_num,  train_accuracy, label="Train accuracy")
        ax2.plot(latent_factors_num,  test_accuracy, color='green', label="Test accuracy")

        handler1, label1 = ax1.get_legend_handles_labels()
        handler2, label2 = ax2.get_legend_handles_labels()

        ax1.legend(handler1+handler2, label1+label2, loc='center right')
        ax1.set_title('Accuracy vs. Number of Latent Factors')
        ax1.grid(linestyle='--')

        ax1.set_xlabel('Number of Latent Factors')
        ax1.set_ylabel('Train accuracy')
        ax2.set_ylabel('Test accuracy', rotation=270, labelpad=12)

        plt.show()
    '''
if __name__ == '__main__':
    from data_clean import Data_Clean
    from mfrecommender import MFRecommender

    #instantiate data clean
    dc = Data_Clean()
    mfr=MFRecommender(dc.articles_clean,dc.interacts_clean)
    latent_factors_num,test_accuracy,train_accuracy =mfr.calculate_error()
    #mfr.draw_curve(latent_factors_num,test_accuracy,train_accuracy)
