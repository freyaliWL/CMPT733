import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

class AnomalyDetection():
    
    def scaleNum(self, df, indices):
        get_len = len(df['features'][0])

        for i in range(get_len):
            if i in indices:
                df['features_'+str(i)] = df['features'].str.get(i)
                df_mean = df['features_'+str(i)].mean()
                df_std = df['features_'+str(i)].std()
                df['features_'+str(i)] = (df['features_'+str(i)]-df_mean)/df_std
                df['features_'+str(i)] = df['features_'+str(i)].apply(lambda x:[x])
            else:
                df['features_'+str(i)] = df['features'].str.get(i)
                df['features_'+str(i)] = df['features_'+str(i)].apply(lambda x:[x])
        
        df_temp = np.empty((len(df), 0))
        df['features'] = df_temp.tolist()

        for i in range(get_len): 
            df['features'] = df['features']+ df['features_'+str(i)]
        df = df[[ "features"]]
        return df



    def cat2Num(self, df, indices):
        
        try:
            df['features'] = df['features'].apply(lambda x:eval(x))
        except:
            pass
        
        get_len = len(df['features'][0])

        for i in range(get_len):
            if i in indices:
                df['features_'+str(i)] = df['features'].apply(lambda x:[x[i]])
                df_sum = np.sum(df['features_'+str(i)])
                unique_entries =  pd.unique(df_sum)
                df['features_'+str(i)] = df['features_'+str(i)].apply(lambda x:[int(j in x) for j in unique_entries]) #check the value #change boolean to int
                
            else:
                df['features_'+str(i)] = df['features'].apply(lambda x:[x[i]])

        df_temp = np.empty((len(df), 0))
        df['features'] = df_temp.tolist()

        for i in range(get_len):
            df['features'] = df['features']+ df['features_'+str(i)]
        df = df[[ "features"]]
        return df


    def detect(self, df, k, t):
        model = KMeans(n_clusters=k)
        f_pre = df['features'].tolist()
        df['clusters'] = model.fit_predict(f_pre) #get predictions
        temp = df.groupby('clusters')
        c_count = temp.count()
        c_max = c_count['features'].max()
        c_min = c_count['features'].min()
        df = c_count.merge(df,on='clusters',how ='right')
        A = c_max-df['features_x']
        B = c_max-c_min
        df['score'] = A/B
        df = df[['features_y','score']]
        df.columns = ['features','score']
        df = df.where(df['score'] >= t).dropna()
        return df



if __name__ == "__main__":
    pd.set_option('mode.chained_assignment', None)

    #df = pd.DataFrame(data=data, columns = ["id", "features"]).set_index('id')
    df = pd.read_csv('logs-features-sample.csv').set_index('id')
    ad = AnomalyDetection()

    df1 = ad.cat2Num(df, [0,1])
    print(df1)

    df2 = ad.scaleNum(df1, [6])
    print(df2)

    df3 = ad.detect(df2, 8, 0.97)
    #df3 = ad.detect(df2, 2, 0.9)
    print(df3)