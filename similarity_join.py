import re
import pandas as pd

class SimilarityJoin:
    def __init__(self, data_file1, data_file2):
        self.df1 = pd.read_csv(data_file1)
        self.df2 = pd.read_csv(data_file2)

    def preprocess_df(self, df, cols): 
        df['joinKey'] = ['']*len(df.index) #get the row count

        for x in cols:
            df['joinKey'] += ' '

            df['joinKey'] += df[x].fillna('')

        expression = lambda x:re.findall(r'\w+', x.lower())
        df['joinKey'] = df['joinKey'].apply(expression)
        
        return df
       

    def filtering(self, df1, df2):
        
        df2 = df2.rename(columns={'id': 'id_2'})
        df1 = df1.rename(columns={'id': 'id_1'})

        df2_n = df2.explode('joinKey')
        df1_n = df1.explode('joinKey')

        #join
        df_m = df1_n.merge(df2_n, on='joinKey')
        df_merged = df_m.drop_duplicates(subset=['id_1', 'id_2'])
        #print(df_merged)
        df2 = df2.rename(columns={'joinKey': 'joinKey_2'})
        df1 = df1.rename(columns={'joinKey': 'joinKey_1'})
        #sub_df on id_1, id_2
        df_selected = df_merged[['id_1', 'id_2']]
        #print(df_selected)
        cand_df = pd.merge(df1, df_selected, on='id_1')
        
        cand_df = pd.merge(df2, cand_df, on='id_2')

        cand_df = cand_df[['id_1', 'joinKey_1', 'id_2', 'joinKey_2']]
        return cand_df
        
    def verification(self, cand_df, threshold):
        jaccard = []
        joinKeys_1 = cand_df['joinKey_1']
        joinKeys_2 = cand_df['joinKey_2']
        len_join = len(joinKeys_1)
        for i in range(len_join):
            set_1 = set(joinKeys_1[i])
            set_2 = set(joinKeys_2[i])
            inter = len(list(set_1&set_2))
            union = len(list(set_1|set_2))
            jaccard.append(inter/union)
        cand_df['jaccard'] = jaccard
        cand_df = cand_df[cand_df['jaccard'] >= threshold]
        return cand_df

    def evaluate(self, ground_truth,result):
        map_g = map(lambda x:x[0],ground_truth)
        map_r = map(lambda x:x[0],result)
        set_r = set(map_r)
        set_g = set(map_g)
        T_l= list(set_g & set_r)
        T = len(T_l)
        precision = T/len(result)
        recall = T/len(ground_truth)
        measure = (precision *recall *2) / (precision + recall)
        return (precision, recall, measure)


    def jaccard_join(self, cols1, cols2, threshold):
        new_df1 = self.preprocess_df(self.df1, cols1)
        new_df2 = self.preprocess_df(self.df2, cols2)
        new_df1.to_csv('df1_temp.csv')
        new_df2.to_csv('df2_temp.csv')
        print ("Before filtering: %d pairs in total" %(self.df1.shape[0] *self.df2.shape[0])) 

        cand_df = self.filtering(new_df1, new_df2)
        print ("After Filtering: %d pairs left" %(cand_df.shape[0]))

        result_df = self.verification(cand_df, threshold)
        print ("After Verification: %d similar pairs" %(result_df.shape[0]))

        return result_df



if __name__ == "__main__":
    ground_truth = pd.read_csv("Amazon_Google_perfectMapping_sample.csv").values.tolist()
    data = SimilarityJoin("Amazon_sample.csv", "Google_sample.csv")
    amazon = ["title", "manufacturer"]
    google = ["name", "manufacturer"]
    df = data.jaccard_join(amazon, google, 0.5)

    result_1 = df[['id_1', 'id_2']].values.tolist()
    result = data.evaluate(ground_truth,result_1)

    print ("(precision, recall, fmeasure) = ", result)