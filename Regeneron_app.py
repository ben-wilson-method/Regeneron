#Import streamlit
import streamlit as st

#Import the packages used
import os
import pickle
import numpy as np
import pandas as pd

#Import the modules for plotting
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt


#Import the NLP packages
import nltk
import string
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#Import the ML Packages
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

#Import the sentence transformers
#from transformers import pipeline
#from sentence_transformers import SentenceTransformer





#Create the sidebar
with st.sidebar:
    #scrapped_website = st.selectbox('Which data would you like to use?',('Reddit',))
    Which_comments = st.selectbox('Select comments to view?',('All','Positive','Negative',))


#Load the data
#if scrapped_website == 'Reddit':
#df = pd.read_csv('Regeneron_scraped_data.csv')
#df_clean = df[df['path'] != '/r/PharmaEire/search/']
#All_comments = []
#for i in range(len(df_clean)):
#    test_string = df_clean['html_content'].to_list()[i]
#    All_comments += [sub_string.split('</p>')[0] for sub_string in test_string.split('<p>')[1:]][4:]
#All_comments = [i.replace('\n','') for i in All_comments]
    
with open('Processed_Data/All_comments.pkl', 'rb') as f:
    All_comments = pickle.load(f)
All_comments = [i.replace('\n','') for i in All_comments]


#Add the sentiments
#Add the model
#sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")

#Create the sentiments
if os.path.isfile('Processed_Data/Sentiments.pkl'):
    with open('Processed_Data/Sentiments.pkl', 'rb') as f:
        Sentiments = pickle.load(f)
else:
    Sentiments = []
    my_bar = st.progress(0.0, text='Creating sentiments')
    for i in range(len(All_comments)):
        my_bar.progress((i+1)/len(All_comments), text='Creating sentiments')
        temp_string = All_comments[i].replace('\n','')
        sentiment = sentiment_analysis(temp_string)
        if sentiment[0]['label'] == 'POSITIVE':
            Sentiments.append(1)
        else:
            Sentiments.append(0)
    my_bar.empty()
    with open('Processed_Data/Sentiments.pkl', 'wb') as f:
        pickle.dump(Sentiments, f)
#Sentiments = []
#for i in range(len(All_comments)):
#    print(str(i+1)+'/'+str(len(All_comments)),end='\r')
#    temp_string = All_comments[i].replace('\n','')
#    sentiment = sentiment_analysis(temp_string)
#    if sentiment[0]['label'] == 'POSITIVE':
#        Sentiments.append(1)
#    else:
#        Sentiments.append(0)    

#Categorise the comments
Pos_comments = []
Neg_comments = []
for i in range(len(All_comments)):
    if Sentiments[i] == 1:
        Pos_comments.append(All_comments[i].replace('\n',''))
    else:
        Neg_comments.append(All_comments[i].replace('\n',''))




#Create the semantic embeddings
if os.path.isfile('Processed_Data/All_Semantic_embeddings.pkl'):
    with open('Processed_Data/All_Semantic_embeddings.pkl', 'rb') as f:
        All_embeddings = pickle.load(f)
    with open('Processed_Data/Pos_Semantic_embeddings.pkl', 'rb') as f:
        Pos_embeddings = pickle.load(f)
    with open('Processed_Data/Neg_Semantic_embeddings.pkl', 'rb') as f:
        Neg_embeddings = pickle.load(f)
else:
    #embedder = SentenceTransformer("all-MiniLM-L6-v2")

    st.write('Creating embeddings')
    All_embeddings = embedder.encode(All_comments)
    Pos_embeddings = embedder.encode(Pos_comments)
    Neg_embeddings = embedder.encode(Neg_comments)
    st.write('Embeddings created')
    with open('Processed_Data/All_Semantic_embeddings.pkl', 'wb') as f:
        pickle.dump(All_embeddings, f)
    with open('Processed_Data/Pos_Semantic_embeddings.pkl', 'wb') as f:
        pickle.dump(Pos_embeddings, f)
    with open('Processed_Data/Neg_Semantic_embeddings.pkl', 'wb') as f:
        pickle.dump(Neg_embeddings, f)


#Do the clustering
gm = GaussianMixture(n_components=6, random_state=42)
All_clust_num = gm.fit_predict(All_embeddings)
gm = GaussianMixture(n_components=6, random_state=42)
Pos_clust_num = gm.fit_predict(Pos_embeddings)
gm = GaussianMixture(n_components=6, random_state=42)
Neg_clust_num = gm.fit_predict(Neg_embeddings)


#Project to a lower space
clf = LDA(n_components=2)

All_lda = clf.fit_transform(All_embeddings, All_clust_num)
Pos_lda = clf.fit_transform(Pos_embeddings, Pos_clust_num)
Neg_lda = clf.fit_transform(Neg_embeddings, Neg_clust_num)


#Plot the results
if Which_comments == 'All':
    st.subheader('All comments')
    st.write('After scraping the Reddit posts and comments relating to Regeneron we embedded them into a semantic space, this means that comments with a similar meaning will be close together and those that are further apart in meaning will be further apart in the embedding space.')
    st.write('We then used a GMM (Gaussian Mixture Model) to group similar comments together. Below is a representation of what this looks like in 2 dimensional space.')
    #fig, ax = plt.subplots()
    #fig.set_figheight(5)
    #fig.set_figwidth(8)
    #ax.scatter(All_lda[:,0],All_lda[:,1],c=All_clust_num)
    #plt.xticks([])
    #plt.yticks([])
    #st.pyplot(plt)

if Which_comments == 'Positive':
    st.subheader('Positive comments')
    st.write('After scraping the Reddit posts and comments relating to Regeneron we used a sentiment analysis model to determine whether the posts and comments were positive or negative in nature.')
    st.write('We then embedded them into a semantic space, this means that comments with a similar meaning will be close together and those that are further apart in meaning will be further apart in the embedding space.')
    st.write('We then used a GMM (Gaussian Mixture Model) to group similar comments together. Below is a representation of what this looks like in 2 dimensional space.')
    #fig, ax = plt.subplots()
    #fig.set_figheight(5)
    #fig.set_figwidth(8)
    #ax.scatter(Pos_lda[:,0],Pos_lda[:,1],c=Pos_clust_num)
    #plt.xticks([])
    #plt.yticks([])
    #st.pyplot(plt)

if Which_comments == 'Negative':
    st.subheader('Negative comments')
    st.write('After scraping the Reddit posts and comments relating to Regeneron we used a sentiment analysis model to determine whether the posts and comments were positive or negative in nature.')
    st.write('We then embedded them into a semantic space, this means that comments with a similar meaning will be close together and those that are further apart in meaning will be further apart in the embedding space.')
    st.write('We then used a GMM (Gaussian Mixture Model) to group similar comments together. Below is a representation of what this looks like in 2 dimensional space.')
    #fig, ax = plt.subplots()
    #fig.set_figheight(5)
    #fig.set_figwidth(8)
    #ax.scatter(Neg_lda[:,0],Neg_lda[:,1],c=Neg_clust_num)
    #plt.xticks([])
    #plt.yticks([])
    #st.pyplot(plt)
    
    


num_clusters = 6

if Which_comments == 'All':
    Embeddings = All_embeddings
    labels = All_clust_num
if Which_comments == 'Positive':
    Embeddings = Pos_embeddings
    labels = Pos_clust_num
if Which_comments == 'Negative':
    Embeddings = Neg_embeddings
    labels = Neg_clust_num
    
    
silhouette_avg = silhouette_score(Embeddings, labels)
silhouette_labels = silhouette_samples(Embeddings, labels)

#Reduce the dimension
clf = LDA(n_components=2)
ans_lda = clf.fit_transform(Embeddings, labels)


fig = go.Figure()

fig.add_trace(
        go.Scatter(
            x=ans_lda[:,0],
            y=ans_lda[:,1],
            mode='markers',
            marker=dict(color=np.array(px.colors.sequential.Plasma)[labels%10],size=10)
            #marker=dict(color=labels,size=10)
        )
    )

fig.update_layout(
    autosize=False,
    width=700,
    height=500,
    margin=dict(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
)

st.plotly_chart(fig)


#Create a plot to show how well the clusters are clustered
st.subheader('Group analysis')
st.write('This is a plot of how well each post / comment fits inside its given group.')

# Create a subplot with 1 row and 2 columns
fig = go.Figure()
named_colorscales = px.colors.named_colorscales()

# The silhouette coefficient can range from -1, 1 but in this example all
# lie within [-0.1, 1]
x_min = -0.1
x_max = 1
# The (n_clusters+1)*10 is for inserting blank space between silhouette
# plots of individual clusters, to demarcate them clearly.
y_min = 0
if Which_comments == 'All':
    y_max = len(All_comments) + (num_clusters + 1) * 10
if Which_comments == 'Positive':
    y_max = len(Pos_comments) + (num_clusters + 1) * 10
if Which_comments == 'Negative':
    y_max = len(Neg_comments) + (num_clusters + 1) * 10

y_lower = 10
for i in range(num_clusters):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    ith_cluster_silhouette_values = silhouette_labels[labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    #color = f'rgb{cm.nipy_spectral(float(i) / num_clusters)[:3]}'
    #color = named_colorscales[0]

    fig.add_trace(go.Scatter(
        x=ith_cluster_silhouette_values,
        y=np.arange(y_lower, y_upper),
        mode='lines',
        fill='tozerox',
        fillcolor=px.colors.sequential.Plasma[i%10],
        line=dict(color=px.colors.sequential.Plasma[i%10],width=0.5),
        showlegend=False
    ))

    # Label the silhouette plots with their cluster numbers at the middle
    fig.add_trace(go.Scatter(
        x=[-0.05],
        y=[y_lower + 0.5 * size_cluster_i],
        text=[str(i+1)],
        mode='text',
        showlegend=False
    ))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

# The vertical line for average silhouette score of all the values
fig.add_shape(
    type="line",
    x0=silhouette_avg, y0=y_min, x1=silhouette_avg, y1=y_max,
    line=dict(color="Red", dash="dash")
)

fig.update_layout(
    #title="The silhouette plot for the various clusters.",
    xaxis=dict(
        title="The silhouette coefficient values",
        range=[x_min, x_max],
        tickvals=[-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1]
    ),
    yaxis=dict(
        title="Cluster label",
        range=[y_min, y_max],
        showticklabels=False
    ),
    width=700,
    height=500
)

st.plotly_chart(fig)






if Which_comments == 'All':
    ans_list_eng = All_comments
if Which_comments == 'Positive':
    ans_list_eng = Pos_comments
if Which_comments == 'Negative':
    ans_list_eng = Neg_comments



#Do the text analysis
ans_clustered = [[] for i in list(set(labels))]
ans_clustered_eng = [[] for i in list(set(labels))]
cluster_metric = [[] for i in list(set(labels))]

for i in list(set(labels)):
    for j in range(len(ans_list_eng)):
        if labels[j] == i:
            #ans_clustered[i].append(ans_list[j])
            ans_clustered_eng[i].append(ans_list_eng[j])
            cluster_metric[i].append(silhouette_labels[j])



n_gram = 2
nltk.download('stopwords')
nltk.download('punkt')

#if Which_comments == 'All':
stop_words = set(stopwords.words('english'))|{'ahahaahah','regeneron','ahahahahah','classrelative','pointereventsauto','nofollow','ugc','relnoopener','rpl','hi'}
#if Which_comments == 'positive':
#    stop_words = 

exclude = set(string.punctuation)|{'・','→'}

#ans_clustered_bog = [[] for i in list(set(labels))]
ans_clustered_eng_bog = [[] for i in list(set(labels))]

for i in range(len(ans_clustered_eng)):
    for j,statement in enumerate(ans_clustered_eng[i]):
        print(str(i+1)+'/'+str(len(ans_clustered_eng))+'-'+str(j+1)+'/'+str(len(ans_clustered_eng[i]))+'          ',end='\r')
        s = ''.join(ch for ch in statement if ch not in exclude)
        word_tokens = word_tokenize(s)
        word_tokens_stopwords = [w.lower() for w in word_tokens if not w.lower() in stop_words]
        filtered_sentence = ngrams(word_tokens_stopwords,n_gram)
        ans_clustered_eng_bog[i] += filtered_sentence

word_counts = []
for i in range(len(ans_clustered_eng)):
    df_tmp = pd.DataFrame(ans_clustered_eng_bog[i])
    df_tmp['words'] = df_tmp[df_tmp.columns].agg(' '.join, axis=1)
    df_tmp = df_tmp[['words']]
    word_counts.append([df_tmp['words'].value_counts()[:10]])


#Give a representation of how accurate each cluster is
st.subheader('Further analysis of the groups')
st.write('The graph below rates the groups on 3 factors, the size of each group, how closely aligned each froup is semantically and the number of commonly occuring words.')


fig = go.Figure()

fig = px.bar(
                x=np.arange(num_clusters)+1,
                #y=[np.mean(i)**(1/3) for i in cluster_metric],
                y=[np.mean(cluster_metric[i])**(1/3)*len(ans_clustered_eng[i])*sum(word_counts[i][0].values)**(1/3) for i in range(len(cluster_metric))],
                #y=[np.mean(cluster_metric[i])*len(ans_clustered_eng[i]) for i in range(len(cluster_metric))],
                color=[np.array(px.colors.sequential.Plasma)[i%10] for i in range(num_clusters)],
                color_discrete_map="identity"
            )

fig.update_layout(
    autosize=False,
    width=700,
    height=500,
    margin=dict(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
)

fig.update_layout(
    #title="The silhouette plot for the various clusters.",
    xaxis=dict(
        title="Cluster label"
    ),
    yaxis=dict(
        title="Cluster rating",
        showticklabels=False
    )
)

st.plotly_chart(fig)

st.write()

cluster_to_inspect = st.selectbox('Which cluster would you like to inspect',(i+1 for i in range(num_clusters)))



word_counts[cluster_to_inspect-1][0]
#word_counts

for statement in ans_clustered_eng[cluster_to_inspect-1]:
    if '<a rpl=' not in statement:
        st.write(statement)
        st.write('####')













