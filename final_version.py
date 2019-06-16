from geolib import geohash
from keras.models import Sequential
from keras.layers import ConvLSTM2D,LSTM,Dense,Bidirectional,Dropout
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


# define the timeseries to supervised learning function
def timeseries_to_supervised(data,geohash_list, n_in=5,n_out=5,col_name='demand'):    
    df = data
    geohash_list=geohash_list 
    append_data=[]
    for geo in geohash_list:
        geo_rows=df[df.geohash6==geo].sort_values(by=['timestamp2'])
        cols, names = list(), list()
        for i in range(n_in, -1, -1):
            cols.append(geo_rows[col_name].shift(i))
            if i==0:
                names += [('%s(t)'%(col_name))]
            else:            
                names += [('%s(t-%d)' % (col_name,i))]
        if n_out != 0:
            cols.append((geo_rows.demand.shift(1)-geo_rows.demand.shift(0))/geo_rows.demand.shift(0))
            names += [('speed1')]
        for i in range(1,n_out+1):
            cols.append(geo_rows.demand.shift(-i))
            names += [('demand(t+%d)'%(i))]

        names +=list(df.columns)
        cols.append(geo_rows)
        geo_rows = pd.concat(cols, axis=1)
        geo_rows.columns = names
        append_data.append(geo_rows)
    df=pd.concat(append_data,axis=0)
    return df
## define the moving average function 
def get_ma (data,geohash_list,period,name):
    df=data
    geohash_list=geohash_list
    cols=list()
    for geo in geohash_list:
        geo_rows=df[df.geohash6==geo].sort_values(by=['timestamp2'])
        geo_rows[name]=geo_rows.demand.rolling(window=period).mean()
        cols.append(geo_rows)
    df=pd.concat(cols,axis=0)
    df.dropna(inplace=True)
    return df
## def the fill mising data function to fill the missing timestamp with 0 demand    
def fill_missing_data(data,geohash_list):
    df=data
    geohash_list=geohash_list 
    cols=list()
    append_data=[]
    for geo in geohash_list:
        geo_rows=df[df.geohash6==geo]
        timestamp2=set(geo_rows.timestamp2)
        timestamp_full=set([i for i in range(0,5856)])
        timestamp_missing=timestamp_full-timestamp2
        for i in list(timestamp_missing):
            missdata=0
            add_row=pd.DataFrame({'geohash6':[geo],
                                  'timestamp2':[i],
                                  'demand':[missdata],
                                  'hrs':['drop']})
            cols.append(add_row)                
        cols.append(geo_rows)
        geo_rows=pd.concat(cols,axis=0)
    append_data.append(geo_rows)
    df=pd.concat(append_data,axis=0).sort_values(by=['timestamp2']).reset_index(drop=True)
    return df[['geohash6','day','hrs','timestamp2','demand']]
    
##get the train or test data for training model
def parse_data (data,n_in=5):
    df=data
    n_in=n_in
    n_out=5
    geohash_list=list(df['geohash6'].unique())
    df['hrs']=df.timestamp.str.split(':').str[0].astype(int)
    df['mins']=df.timestamp.str.split(':').str[1].astype(int)
    df['timestamp2']=(df.day-1)*96+df.hrs*4+df.mins/15
    df['timestamp2']=df['timestamp2'].astype(int)
    ##call the function to fill missing timestamp with deomand 0
    df=fill_missing_data(df,geohash_list)
    ##call the function to get the moving average demand
    feature2_period,feature2_name=48,'%sma_demand'%(48)
    feature3_period,feature3_name=96,'%sma_demand'%(96)
    df=get_ma(df,geohash_list,feature2_period,feature2_name)
    df=get_ma(df,geohash_list,feature3_period,feature3_name)
    ##call the function to transfer timing series data to supervised learning problem
    train_test_data=timeseries_to_supervised(df,geohash_list,n_in,n_out)
    train_test_data=timeseries_to_supervised(train_test_data,geohash_list,n_in,0,col_name=feature2_name)
    train_test_data=timeseries_to_supervised(train_test_data,geohash_list,n_in,0,col_name=feature3_name)
    train_test_data=train_test_data[train_test_data.hrs!='drop']
    train_test_data.dropna(inplace=True)
    return train_test_data

n_in=5
n_out=5
##get the train data
df_train=pd.read_csv("../input/training.csv")
train_data=parse_data (df_train,n_in)
feature_size=3*(n_in+1)+1
train_X=train_data.iloc[:,0:feature_size]
train_y=train_data.iloc[:,feature_size:(feature_size+n_out)]
## get the test data
df_test=pd.read_csv("../input/testing.csv")
test_data=parse_data(df_test,n_in)
test_X = test_data.iloc[:,0:feature_size]
test_y = test_data.iloc[:,feature_size:(feature_size+n_out)]
##reshape the train_X and test_X shape
train_X = train_X.values.reshape((train_X.shape[0], 1,train_X.shape[1]))
test_X_reshape = test_X.values.reshape((test_X.shape[0], 1,test_X.shape[1]))
##build LSTM model
model = Sequential()
model.add(LSTM(128, activation='tanh', return_sequences=True, input_shape=(1, feature_size)))
model.add(LSTM(50, activation='tanh'))
model.add(Dense(5))
model.compile(optimizer='adam', loss='mse')
##train model
trained_model=model.fit(train_X, train_y.values, epochs=50, batch_size=300,verbose=0)
plt.plot(trained_model.history['loss'])
## predict
yhat = model.predict(test_X_reshape, verbose=0)
##evaluation
print (test_y.values)
print (yhat)
RMSE=np.sqrt(mean_squared_error(test_y.values, yhat))
print (RMSE)
## plot yhat and test y
plt.figure(figsize=(20,10))
plt.plot(yhat[:,0])
plt.plot(test_y.values[:,0])