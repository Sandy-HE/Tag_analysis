library("data.table")
library(text2vec)
library(vegan)

cleantag_df1 <- fread("cleantaglv1.csv")
cleantag_df2 <- fread("cleantaglv2.csv")



##single word tokenizer
tok_fun = word_tokenizer

#when train for cleantag_df2, just change the variables
it_train = itoken(cleantag_df1$cleanlv1, 
                  #preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  ids = cleantag_df1$track_id, 
                  progressbar = T)

#tokens <- space_tokenizer(train$fulltext, sep=";")
#it_train <- itoken(tokens, 
#                   ids = train$track_id,   
#                   progressbar = T) 


system.time(vocab <- create_vocabulary(it_train))

##remove low-freq and high-freq words
pruned_vocab = prune_vocabulary(vocab,   
                                term_count_min = 1000,   
                                doc_proportion_max = 0.8,  
                                doc_proportion_min = 0.0002)  

##set up corpus vector
vectorizer <- vocab_vectorizer(pruned_vocab)

##generate DTM(doc-term matrix). Here we can understand it as song-term matrix
system.time(dtm_train <- create_dtm(it_train, vectorizer))


##generate TCM(term-co-occurrence matrix).
system.time(tcm_train <- create_tcm(it_train, vectorizer, skip_grams_window = 5L,
                                    skip_grams_window_context = c("symmetric", "right", "left")))


#====Glove model===
#This model will use tcm as input, output is word vectors
#You can adjust vector size
glove = GlobalVectors$new(word_vectors_size = 150, vocabulary = pruned_vocab, x_max = 10, learning_rate=0.15)
wv_main = glove$fit_transform(tcm_train, n_iter = 25, convergence_tol = 0.005)
wv_context = glove$components
word_vectors = wv_main + t(wv_context)


#Save word vectors if neccesary, because when you change vector size, the word vector is changed time by time. 
#If you want to get stable result. Just save your result.
wordvec_df <- as.data.frame(word_vectors)
wordvec_df$term <- rownames(word_vectors)
#fwrite(as.data.frame(word_vectors), "wordvec_<size>.csv")
fwrite(wordvec_df, "lv1_glove_d150_wordvec.csv")
#fwrite(wordvec_df, "lv2_glove_d150_wordvec.csv")
#fwrite(wordvec_df, "phrase_glove_d150_wordvec.csv")

#read any back up data
#wordvec_df <- fread("lv1_glove_d150_wordvec.csv")
#word_vectors <- as.matrix(wordvec_df[,1:150])
#rownames(word_vectors) <- wordvec_df$term

eterms_df <- fread("./emotion_terms_list.csv")
sub_wordvec <- subset(word_vectors, rownames(word_vectors) %in% eterms_df$term)
sub_wordvec <- sub_wordvec[order(rownames(sub_wordvec)),]

#====This part is for data exploration(optional)====
#apply different dimensionality reduction methods to D150 data to check which method is best
library(dimRed)
drdata <- dimRedData(data = sub_wordvec, meta = rownames(sub_wordvec))
#drdata <- dimRedData(data = wordvec_df[,1:150], meta = wordvec_df$term)
embed_methods <- c("Isomap", "PCA","MDS","LLE","nMDS","kPCA")
# ,"AutoEncoder","tSNE","UMAP")
#AutoEncoder and UMAP use python backend, tSNE shows perplexity error
data_emb <- lapply(embed_methods, function(x) embed(drdata, x))
names(data_emb) <- embed_methods
plot_R_NX(data_emb)
#====End of this part=====

#Obtain ground truth data for valence-arousal-dominant rating
#Warriner VAD rating, value range is [1,9] 
baseline_df = fread("Warriner_avd_ratings.csv")
baseline_df = baseline_df[,c("Word","V.Mean.Sum","A.Mean.Sum","D.Mean.Sum")]
baseline_df = baseline_df[baseline_df$Word %in% rownames(sub_wordvec),]
#Normalized to [-1,1]
baseline_mt = as.matrix(baseline_df[,2:3])
baseline_mt= (baseline_mt-1)*2/8-1
rownames(baseline_mt) <- baseline_df$Word
#Generate cosine similarity for baseline word vectors
#basesim_mt <- lsa::cosine(t(baseline_mt))
#Convert matrix to vector, so that we can evaluate it in gof() function
#basesim_vec <- basesim_mt[upper.tri(basesim_mt)]


#nMDS -- reduce to 2D or 3D word vectors
#calculate pairwise-rows cosine similarity and generate a similarity matrix
library(vegan)
termsim_mt <-lsa::cosine(t(sub_wordvec))
termdis_mt <- max(termsim_mt)-termsim_mt

mds_model<-metaMDS(termdis_mt, k=3)
#mds_model<-metaMDS(termdis_mt, k=2)
#stressplot(mds_model,termdis_mt)
mds_terms <- mds_model$points
mds_terms <- mds_terms[rownames(mds_terms) %in% baseline_df$Word ,]
#mds_terms_exchange <- mds_terms[,c("MDS2","MDS1")]
#termsim_mt_3d <- lsa::cosine(t(mds_terms))
#termsim_mt_2d <- lsa::cosine(t(mds_terms))

#trailor upper diagonal
#termsim_vec <- termsim_mt_3d[upper.tri(termsim_mt_3d)]

#termsim_vec_150 <- sub_wordvec[rownames(sub_wordvec) %in% baseline_df$Word,]
#termsim_vec_150 <- termsim_vec_150[order(rownames(termsim_vec_150)),]
#termsim_vec_150 <- lsa::cosine(t(termsim_vec_150))
#termsim_vec_150 <- termsim_vec_150[upper.tri(termsim_vec_150)]

#isomap_model
#value range is out of [-1,1]
#isomap_model <- isomap(termdis_mt, ndim=3,k=3)
#isomap_terms <- isomap_model$points
#isomap_terms <- isomap_terms[rownames(isomap_terms) %in% baseline_df$Word,]
#termsim_mt_2d <- lsa::cosine(t(isomap_terms))
#plot(isomap_model)

#library(hydroGOF)
#goodness of fit
#gof(termsim_vec,basesim_vec)

plot(mds_terms, type= 'n')
text(mds_terms,rownames(mds_terms),cex=.7)

plot(baseline_mt, type= 'n')
text(baseline_mt,rownames(baseline_mt),cex=.7)

#Evaluate word embedding performance
proc <- procrustes(baseline_mt,mds_terms)
plot(proc)
summary(proc)
#residuals(proc)

