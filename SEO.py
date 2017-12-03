import os
import math
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus_root='D:/sem2/5334_Data_Mining/PA1/presidential_debates'
stemmer = PorterStemmer()
file_names=[]
file_paths=[]
sw=stopwords.words("english")
words=[]
tokens=[]
stem1=[]
query_words=[]
query_tokens=[]
query_tokens_stem=[]
tokens_stem_per_file=[]
count_terms_per_doc={}
document_frequency_list={}
log_freq_weight_matrix={}
idf_list={}
tfidf_weight_matrix={}
query_count_list={}
query_log_freq_weight_matrix={}
query_tfidf_weight_matrix={}
count_query={}
query1_tfidf={}
doc1_sqrt=0
max_list=[]
max_no=0
count_query_term={}
query2_tfidf={}
query_doc_cos_sim={}
match_doc=""
def get_all_words():
    for filename in os.listdir(corpus_root):
        file = open(os.path.join(corpus_root, filename), "r", encoding='UTF-8')

        file_names.append(filename)
        file_paths.append(os.path.join(corpus_root, filename))
        doc = file.read()

        doc = doc.lower()
        words.extend(doc.split(" "))
    return words

def get_tokens_stem():
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    for i in words:
        tokens.extend(tokenizer.tokenize(i))
    for i in tokens:
        if i not in sw:
            stem1.append(stemmer.stem(i))
    return stem1

def create_tokens_stem(t):
    query_tokens=[]
    query_words=t.split(" ")
    for qwords in query_words:
        tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
        query_tokens.extend(tokenizer.tokenize(qwords))
    return query_tokens

def remove_sw(t):
    query_tokens_stem=[]
    query_tokens1=[]
    query_tokens1=create_tokens_stem(t)
    for qtokens in query_tokens1:
        if qtokens not in sw:
            query_tokens_stem.append(stemmer.stem(qtokens))
    return query_tokens_stem


def count_each_doc():
    words_per_doc={}
    for i in range(len(file_paths)):
        file=open(file_paths[i], "r", encoding='UTF-8')
        doc1=file.read()
        doc1=doc1.lower()
        words_per_doc={}
        for words in remove_sw(str(doc1)):
            if words not in words_per_doc:
                words_per_doc[words]=1
            else:
                words_per_doc[words]+=1
        for l in stem1:
            if l not in words_per_doc:
                words_per_doc[l]=0
            count_terms_per_doc[file_names[i]]=words_per_doc
    return count_terms_per_doc



def log_freq_weight():
    for i in count_terms_per_doc:
        log_freq_weight_matrix[i]={}
        for j in count_terms_per_doc[i]:
            if count_terms_per_doc[i][j]>0:
                log_freq_weight_matrix[i][j]=1+math.log(count_terms_per_doc[i][j],10)
            else:
                log_freq_weight_matrix[i][j]=0
    return log_freq_weight_matrix

def document_frequency():
    count1=0
    for i in stem1:
        document_frequency_list[i]=0
    for j in count_terms_per_doc:
        for k in count_terms_per_doc[j]:
            for l in document_frequency_list:
                if l==k and count_terms_per_doc[j][k]!=0:
                    document_frequency_list[l]+=1
    return document_frequency_list

def idf():
    N=len(file_names)
    for i in document_frequency_list:
        idf_list[i]=abs(math.log((N/document_frequency_list[i]),10))
    return idf_list

def tfidf_weight():
    for i in count_terms_per_doc:
        tfidf_weight_matrix[i]={}
        for j in count_terms_per_doc[i]:
            tfidf_weight_matrix[i][j]=log_freq_weight_matrix[i][j]*idf_list[j]
    return tfidf_weight_matrix

def getcount(t):
    count=0
    h=remove_sw(t)
    for j in count_terms_per_doc:
        for k in count_terms_per_doc[j]:
            if k==h[0]:
                count=count+ count_terms_per_doc[j][k]
    return count

def getidf(t):
    idf=0
    t_sw=remove_sw(t)
    N=len(file_names)
    for i in document_frequency_list:
        if t_sw[0]==i:
            idf=abs(math.log(document_frequency_list[i]/N,10))
    return idf

def query_count_matrix(q):
    q_sw=[]
    q_sw=remove_sw(q)
    for i in q_sw and stem1:
        query_count_list[i]=0
    for i in q_sw:
        if query_count_list[i]==0:
            query_count_list[i]=1
        elif query_count_list[i]>0:
            query_count_list[i]+=1
    return query_count_list


def query_log_freq_weight():
    query_log_freq_weight_matrix=query_count_list
    for i in query_log_freq_weight_matrix:
        if query_log_freq_weight_matrix[i]>0:
            query_log_freq_weight_matrix[i]=1+math.log(query_log_freq_weight_matrix[i],10)
        else:
            query_log_freq_weight_matrix[i]=0
    return query_log_freq_weight_matrix


def docdocsim(f1,f2):
    cos_doc_doc_summ=0
    doc1_summ=0
    doc2_summ=0

    doc2_sqrt=0
    doc_doc_cos_sim=0
    doc_doc_list=[f1,f2]
    count=0
    for i in tfidf_weight_matrix:
        count+=1
        for j in tfidf_weight_matrix[i]:
            cos_doc_doc_summ+=tfidf_weight_matrix[f1][j]*tfidf_weight_matrix[f2][j]
            doc1_summ+=(tfidf_weight_matrix[f1][j]*tfidf_weight_matrix[f1][j])
            doc2_summ+=(tfidf_weight_matrix[f2][j]*tfidf_weight_matrix[f2][j])
    doc1_sqrt=math.sqrt(doc1_summ/count)
    doc2_sqrt=math.sqrt(doc2_summ/count)
    doc_doc_cos_sim=(cos_doc_doc_summ/count)/(doc1_sqrt*doc2_sqrt)
    return doc_doc_cos_sim

def querydocsim(q,d):
    count_query=query_count_matrix(q)
    query1_tfidf=query_log_freq_weight()
    cos_query_doc_summ=0
    query_summ=0
    doc_summ=0
    query_sqrt=0
    doc_sqrt=0
    cos_query_doc_sim=0
    for j in tfidf_weight_matrix[d]:
        cos_query_doc_summ+=query1_tfidf[j]*tfidf_weight_matrix[d][j]
        doc_summ+=tfidf_weight_matrix[d][j]*tfidf_weight_matrix[d][j]
        query_summ+=query1_tfidf[j]*query1_tfidf[j]
    query_sqrt=math.sqrt(query_summ)
    doc_sqrt=math.sqrt(doc_summ)
    cos_query_doc_sim = cos_query_doc_summ/(query_sqrt*doc_sqrt)
    return cos_query_doc_sim

def query(que):
    count_query_term=query_count_matrix(que)
    query2_tfidf=query_log_freq_weight()
    count1=0
    max_sim=0
    for i in tfidf_weight_matrix:
        count1+=1
        query_doc_cos_sim[i]=0
    for i in tfidf_weight_matrix:
        qd_summ=0
        d_summ=0
        q_summ=0
        for j in tfidf_weight_matrix[i]:
            qd_summ+=query2_tfidf[j]*tfidf_weight_matrix[i][j]
            d_summ+=tfidf_weight_matrix[i][j]*tfidf_weight_matrix[i][j]
            q_summ+=query2_tfidf[j]*query2_tfidf[j]
        d_sqrt=math.sqrt(d_summ)
        q_sqrt=math.sqrt(q_summ)
        cos_sim=(qd_summ)/d_sqrt*q_sqrt
        query_doc_cos_sim[i]=cos_sim
    for i in query_doc_cos_sim:
        if query_doc_cos_sim[i] > max_sim :
            max_sim=query_doc_cos_sim[i]
    for i in query_doc_cos_sim:
        if query_doc_cos_sim[i] == max_sim:
            match_doc=i
    return match_doc




class SEO:
    get_all_words()
    get_tokens_stem()
    count_each_doc()
    print(getcount('attack'))
    document_frequency()
    print("%.12f" % getidf("agenda"))
    log_freq_weight()
    idf()
    tfidf_weight()
    query_log_freq_weight()
    print("%.12f" % docdocsim("1960-10-21.txt", "1980-09-21.txt"))
    print("%.12f" %querydocsim("particular constitutional amendment", "2000-10-03.txt"))
    print(query("particular constitutional amendment"))