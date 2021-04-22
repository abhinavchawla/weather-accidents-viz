import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
from flask import Flask, jsonify
from flask import render_template
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

app = Flask(__name__)
df=pd.read_csv('data/final.csv')


df_numerical=df.drop(['Nationality', 'Age'], axis=1)


scaler = StandardScaler()
scaled_df=df_numerical.copy()
scaled_df=pd.DataFrame(scaler.fit_transform(scaled_df), columns=scaled_df.columns)

kmeans=KMeans(n_clusters=3)
kmeans.fit(scaled_df)
labels=kmeans.labels_.tolist()

pca = PCA(n_components=18)
principalComponents = pca.fit_transform(scaled_df)
minmax=MinMaxScaler(feature_range=(-1,1))
principalComponents=minmax.fit_transform(principalComponents)
PC1=principalComponents[:,0].tolist()
PC2 = principalComponents[:, 1].tolist()

components = ['PC 1', 'PC 2','PC 3','PC 4','PC 5','PC 6','PC 7','PC 8','PC 9','PC 10','PC 11','PC 12','PC 13','PC 14','PC 15','PC16','PC 17','PC 18']
exp_var=pca.explained_variance_ratio_
out_sum = np.cumsum(pca.explained_variance_ratio_)
res = dict(zip(components, exp_var))


def get_pcp_data():
    attributes=['Nationality','Overall','Aggression','Balance', 'Ball control','Dribbling', 'Finishing', 'Interceptions','Jumping','Age']
    df_pcp=df[attributes]
    data=df_pcp.values
    pcp_list=[]
    i=0
    for val in data:
        tmp_dic={}
        tmp_dic[attributes[0]]=val[0]
        tmp_dic[attributes[1]]=val[1]
        tmp_dic[attributes[2]]=val[2]
        tmp_dic[attributes[3]]=val[3]
        tmp_dic[attributes[4]]=val[4]
        tmp_dic[attributes[5]]=val[5]
        tmp_dic[attributes[6]]=val[6]
        tmp_dic[attributes[7]]=val[7]
        tmp_dic[attributes[8]]=val[8]
        tmp_dic[attributes[9]]= str(val[9])
        tmp_dic['label']=labels[i]
        i+=1
        pcp_list.append(tmp_dic)
    return pcp_list


def get_list1():
    list1 = []
    for key, val in res.items():
        list1.append({'component': key, 'explained_variance': val})
    i=0
    for val in list1:
        val['cumulative_variance'] = out_sum[i]
        i += 1
    return list1

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loading_matrix = pd.DataFrame(loadings, columns=components, index=df_numerical.columns)

attr=loading_matrix.index

def get_mds():
    mds = MDS(dissimilarity='euclidean')
    #mds_df = df_numerical.sample(50)
    mds_df=mds.fit_transform(df_numerical)
    mds_list=[]
    i=0;
    for val in mds_df:
        tmp_dic={}
        tmp_dic['x']=val[0]
        tmp_dic['y']=val[1]
        tmp_dic["label"]=labels[i]
        i+=1
        mds_list.append(tmp_dic)
    return mds_list


def get_variables_mds():
    mds=MDS(dissimilarity='precomputed')
    attributes=df_numerical.columns
    corr_df=df_numerical.corr()
    var_mds_list=[]
    corr_df=corr_df.abs()
    corr_df=corr_df*-1
    corr_df=corr_df+1
    variable_mds=mds.fit_transform(corr_df)
    i=0
    for val in variable_mds:
        tmp_dic={}
        tmp_dic['x']=val[0]
        tmp_dic['y']=val[1]
        tmp_dic['attr']=attributes[i]
        i+=1
        var_mds_list.append(tmp_dic)
    return(var_mds_list)


def get_biplot_vector():
    biplot_vector=[]
    for val in attr:
        tmp_pc1=loading_matrix.loc[val,'PC 1']
        tmp_pc2 = loading_matrix.loc[val, 'PC 2']
        biplot_vector.append({'attr':val,'pc1':tmp_pc1,'pc2':tmp_pc2})
    return biplot_vector


def getMaxAtributes(int_dim_idx):
    tmp_dic={}
    for val in attr:
        sqr_load_val=0
        for x in range(int_dim_idx):
            sqr_load_val+=loading_matrix.loc[val,components[x]]**2
        tmp_dic[val]=sqr_load_val

    max_attr=[]

    for k in sorted(tmp_dic, key=tmp_dic.get, reverse=True):
        max_attr.append(k)
        if len(max_attr)==4:
            break
    return max_attr

def getMatrixData(int_dim_idx):
    scatter_mat = []

    max_attr=getMaxAtributes(int_dim_idx)

    j=0
    attr_vals=scaled_df.loc[:,max_attr].values
    for val in attr_vals:
        my_dic={}
        i=0
        for elem in val:
            my_dic[max_attr[i]]=elem
            i+=1
        my_dic['label']=labels[j]
        j+=1
        scatter_mat.append(my_dic)

    return scatter_mat

def get_loading_table(int_dim_idx):
    table_list=[]
    max_attr=getMaxAtributes(int_dim_idx)
    for elem in max_attr:
        tmp_dic={'Attribute':elem}
        for val in range(int_dim_idx):
            tmp_dic[components[val]]=loading_matrix.loc[elem,components[val]]
        table_list.append(tmp_dic)
    return table_list


@app.route("/pcp")
def pcp_data():
    pcp=get_pcp_data()
    print(pcp)
    return jsonify(pcp)

@app.route("/variable_mds")
def get_variable_mds_data():
    var_mds=get_variables_mds()
    return jsonify(var_mds)


@app.route("/loading_table/<idx>")
def loading_table(idx):
    dim_idx = int(idx) + 1
    table=get_loading_table(dim_idx)
    print(table)
    return jsonify(table)

@app.route("/bar")
def bar():
    myList=get_list1()
    return jsonify(myList)

@app.route("/biplot")
def biplot():
    msg=[]
    for val in PC1:
        msg.append({'PC1':val})
    i=0
    for val in msg:
        val['PC2']=PC2[i]
        i+=1
    return jsonify(msg)

@app.route("/biplot_vector")
def inside_biplot():
    vect=get_biplot_vector()
    return jsonify(vect)

@app.route("/matrix_scatterplot/<idx>")
def create_matrix_scat(idx):
    dim_idx=int(idx)+1
    mat=getMatrixData(dim_idx)
    print(mat)
    return jsonify(mat)

@app.route("/mds")
def get_mds_data():
    mds_data=get_mds()
    return jsonify(mds_data)

@app.route("/")
def d3_main():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)