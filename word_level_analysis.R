library("data.table")
library("text2vec")
library("plotrix")
library("tidyverse")


cleantag_df1 <- fread("cleantaglv1.csv")
cleantag_df2 <- fread("cleantaglv2.csv")
cleantag_df3 <- fread("cleantaglv3.csv")


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
                                #term_count_max = 30000)  

##set up corpus vector
vectorizer <- vocab_vectorizer(pruned_vocab)

##generate DTM(doc-term matrix). Here we can understand it as song-term matrix
system.time(dtm_train <- create_dtm(it_train, vectorizer))


##generate TCM(term-co-occurrence matrix).
system.time(tcm_train <- create_tcm(it_train, vectorizer, skip_grams_window = 5L,
                                    skip_grams_window_context = "right"))

temp=as.matrix(tcm_train)
temp1=temp-diag(diag(temp))

temp2 = normalize(tcm_train, "l2")
#temp2=as.matrix(temp2)
#====Glove model===
#This model will use tcm as input, output is word vectors
#You can adjust vector size
#glove = GlobalVectors$new(word_vectors_size = 150, vocabulary = pruned_vocab, x_max = 10, learning_rate=0.15)
#wv_main = glove$fit_transform(tcm_train, n_iter = 25, convergence_tol = 0.005)
glove = GlobalVectors$new(rank = 4, x_max = 10)
wv_main = glove$fit_transform(tcm_train, n_iter = 25, convergence_tol = 0.001, n_threads = 4)
wv_context = glove$components
word_vectors = wv_main + t(wv_context)


#Save word vectors if neccesary, because when you change vector size, the word vector is changed time by time. 
#If you want to get stable result. Just save your result.
wordvec_df <- as.data.frame(word_vectors)
wordvec_df$term <- rownames(word_vectors)
#fwrite(as.data.frame(word_vectors), "wordvec_<size>.csv")
fwrite(wordvec_df, "phrase_7685_right_glove_D64.csv")
#fwrite(wordvec_df, "lv3_8047_right_glove_D4.csv")
#fwrite(wordvec_df, "phrase_glove_d150_wordvec.csv")
temp3= wordvec_df[wordvec_df$term %in% c("happy","sad"),]

#read any back up data
#wordvec_df <- fread("phrase_7685_LSA_D128.csv")
wordvec_df <- fread("phrase_7685_right_glove_D64.csv")
#wordvec_df <- fread("lv1_8100_right_glove_D4.csv")
#wordvec_df <- fread("lv3_8047_right_glove_D64.csv")
wordvec_df <- fread("phrase_emotion_only_LSA_D32.csv")
word_vectors <- as.matrix(wordvec_df[,1:32])
rownames(word_vectors) <- wordvec_df$term


eterms_df <- fread("./emotion_terms_list3.csv")
sub_wordvec <- subset(word_vectors, rownames(word_vectors) %in% eterms_df$term)
sub_wordvec <- sub_wordvec[order(rownames(sub_wordvec)),]

## Compute Hopkins statistic
##method-1, the result is not stable
#library(clustertend)
#hopkins(sub_wordvec,n=4)

##method-2
#library(factoextra)
# res <- get_clust_tendency(sub_wordvec, n = 4, graph = TRUE)
# res$hopkins_stat
# res$plot

## K-means and visualization
# km.res1 <- kmeans(sub_wordvec, 4)
# fviz_cluster(list(data = sub_wordvec, cluster = km.res1$cluster),
#              ellipse.type = "norm", geom = "point", stand = FALSE,
#              palette = "jco", ggtheme = theme_classic())

#====This part is for data exploration(optional)====
#apply different dimensionality reduction methods to D150 data to check which method is best
library(dimRed)
#library(reticulate)
library(tensorflow)
#tf version 2.0 dose not work, because autoencoder use placeholder, but tfv2.0 remove it.
#install_tensorflow(method = "auto",version=1.14)
#tensorflow::tf_version()

drdata <- dimRedData(data = sub_wordvec, meta = rownames(sub_wordvec))
#drdata <- dimRedData(data = wordvec_df[,1:150], meta = wordvec_df$term)
embed_methods <- c("PCA","LLE","nMDS","kPCA","AutoEncoder")
# "Isomap","AutoEncoder","tSNE","UMAP")
#AutoEncoder and UMAP use python backend, tSNE shows perplexity error
#UMAP use python pacakge "umap-learn"
data_emb <- lapply(embed_methods, function(x) embed(drdata, x))
names(data_emb) <- embed_methods
plot_R_NX(data_emb)
quality_methods <- c("Q_local", "Q_global", "AUC_lnK_R_NX", "cophenetic_correlation")
qual <- sapply(data_emb, function(x) quality(x, "cophenetic_correlation"))
#====End of this part=====

#Obtain ground truth data for valence-arousal-dominant rating
#Warriner VAD rating, value range is [1,9] 
baseline_df = fread("Warriner_avd_ratings.csv")
baseline_df = baseline_df[,c("Word","V.Mean.Sum","A.Mean.Sum","D.Mean.Sum")]
baseline_df = baseline_df[baseline_df$Word %in% rownames(sub_wordvec),]
#baseline_df = baseline_df[baseline_df$Word %in% eterms_df$term,]
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
library(lsa)

termsim_mt <-lsa::cosine(t(sub_wordvec))
termdis_mt <- max(termsim_mt)-termsim_mt

mds_model<-metaMDS(termdis_mt, k=3)
mds_terms <- mds_model$points
mds_terms <- mds_terms[rownames(mds_terms) %in% baseline_df$Word ,]
proc <- procrustes(baseline_mt,mds_terms)
#plot(proc)
summary(proc)

mds_model<-metaMDS(termdis_mt, k=2)
#stressplot(mds_model,termdis_mt)
mds_terms <- mds_model$points
mds_terms <- mds_terms[rownames(mds_terms) %in% baseline_df$Word ,]
#mds_terms_exchange <- mds_terms[,c("MDS2","MDS1")]
proc <- procrustes(baseline_mt,mds_terms)
#plot(proc)
summary(proc)

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
residuals(proc)

#scale rotated result same as the scale of baseline. Here baseline scale is [-0.8,0.8]
temp= proc$Yrot
proc$Yrot= (temp+0.45)*1.6/0.9-0.8   #for skip-gram scale 0.45
#proc$Yrot= (temp+0.5)*1.6/1-0.8   #for glove scale 0.5
#proc$Yrot= (temp+0.6)*1.6/1.2-0.8     #for emotion only (lsa, cbow, sg), 0.6

plot.new()
plot(c(-1.1,1.1), c(-1.1,1.1), type='n', asp=1,xlab = "(b)", ylab = "")
draw.circle(0, 0, 1, nv = 1000, border = NULL, col = NA, lty = 1, lwd = 1)
arrows(c(-1.1,0),c(0,-1.1),c(1.1,0),c(0,1.1), length=0.1)

points(proc$Yrot,pch=16, col="blue")
text(proc$Yrot,adj=c(0,1),rownames(proc$Yrot), cex = .7)
#newdata <- predict(proc,target_ex)
#points(newdata, pch=8, col="green")
#text(newdata,adj=c(0,1),rownames(newdata), cex=.7)
points(baseline_mt,pch=1, col="red")
text(baseline_mt, adj=c(0,1), rownames(baseline_mt),cex=.7, col= "grey")
arrows(proc$Yrot[,1],proc$Yrot[,2], baseline_mt[,1],baseline_mt[,2],length=0.1, col="grey")

text(x=0.22,y=1.1, "Arousal", font=2)
text(x=1.2,y=0.1, "Valence", font=2)
text(x=1.15,y=-0.05, "positive" , cex=.7, color="grey", font=3)
text(x=-1.15,y=-0.05, "negative" , cex=.7, color="grey", font=3)
text(x=-0.15,y=1.05, "active" , cex=.7, color="grey", font=3)
text(x=-0.15,y=-1.05, "inactive" , cex=.7, color="grey", font=3)



#====correlation measurement====
#cor.test(baseline_mt,mds_terms,method = "pearson")
#cor.test(x=, y=, method = 'spearman')

#====confusion matrix====
library(caret)
pred = as.data.frame(proc$Yrot)
pred$term=rownames(pred)
pred= pred %>%
  mutate(quadrant = case_when(V1 > 0 & V2 > 0 ~ 1,
                              V1 < 0 & V2 > 0 ~ 2,
                              V1 < 0 & V2 < 0 ~ 3,
                              V1 > 0 & V2 < 0 ~ 4))
pred$quadrant=as.factor(pred$quadrant)
truth = eterms_df[order(eterms_df$term),]
truth$quadrant=as.factor(truth$quadrant)

confusionMatrix(
  pred$quadrant,
  truth$quadrant,
  pred$term
)


#====performance visualization of glove and lsa====
library(ggplot2)
library(tidyverse)
#library(hrbrthemes)

# Load dataset
performdata <- fread("glove_lsa_comparison_list3_44terms.csv")
performdata$K=factor(performdata$K,levels=c("D4","D8","D16","D32","D64","D128"),ordered=FALSE)
performdata$K=factor(performdata$K,levels=c("4","8","16","32","64","128"),ordered=FALSE)

performdata_new = performdata %>%
  gather("method","perform","GloVe_kD+MDS_2D":"Skip-gram_kD+MDS_3D")
# Plot fill="#69b3a2"
performdata_new %>%
ggplot(aes(x=K, y=perform, group=method)) +
  geom_point(aes(color=method,shape=method),size=3) +
  geom_line(aes(color=method)) +
  scale_color_brewer(palette="Dark2")+
  theme_bw()+
  #xlab("Dimension Reduced")+
  ylab("Procrustes Analysis RMSE")+
  #scale_y_continuous(limits = c(0, 1))
  coord_cartesian(ylim = c(0.5, 0.7))+
  theme(legend.title = element_blank(), 
        legend.key.width=unit(1, "cm"),
        legend.text=element_text(size=12),
        axis.title.y=element_text(size=15),
        axis.title.x=element_text(size=15),
        axis.text=element_text(size=15),
        legend.position=c(0.8, 0.78))

