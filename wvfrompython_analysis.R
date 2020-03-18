library("data.table")
library(vegan)


word2vec_df <- fread("eterm89_word2vec_D150_cbow.csv")
word2vec_df <- fread("eterm89_word2vec_D150_sg_iter30.csv")
word2vec_df <- fread("eterm89_word2vec_D100_sg.csv")

word_vectors <- as.matrix(word2vec_df[,2:151])
rownames(word_vectors) <- word2vec_df$term
word_vectors <- word_vectors[order(rownames(word_vectors)),]

#====This part is for data exploration(optional)====
#apply different dimensionality reduction methods to D150 data to check which method is best
library(dimRed)
drdata <- dimRedData(data = word_vectors, meta = rownames(word_vectors))
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
baseline_df = baseline_df[baseline_df$Word %in% rownames(word_vectors),]
#Normalized to [-1,1]
baseline_mt = as.matrix(baseline_df[,2:3])
baseline_mt= (baseline_mt-1)*2/8-1
rownames(baseline_mt) <- baseline_df$Word
#Generate cosine similarity for baseline word vectors
#basesim_mt <- lsa::cosine(t(baseline_mt))
#Convert matrix to vector, so that we can evaluate it in gof() function
#basesim_vec <- basesim_mt[upper.tri(basesim_mt)]

eterm_left <- baseline_df$Word
#nMDS -- reduce to 2D or 3D word vectors
#calculate pairwise-rows cosine similarity and generate a similarity matrix

termsim_mt <-lsa::cosine(t(word_vectors))
termdis_mt <- max(termsim_mt)-termsim_mt

mds_model<-metaMDS(termdis_mt, k=2)
#mds_model<-metaMDS(termdis_mt, k=2)
#stressplot(mds_model,termdis_mt)
mds_terms <- mds_model$points
mds_terms <- mds_terms[rownames(mds_terms) %in% baseline_df$Word ,]
#mds_terms_exchange <- mds_terms[,c("MDS2","MDS1")]

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

#plot(baseline_mt, type= 'n')
#text(baseline_mt,rownames(baseline_mt),cex=.7)

#Evaluate word embedding performance
proc <- procrustes(baseline_mt,mds_terms)
plot(proc)
summary(proc)
#residuals(proc)

eterm_classes <- kmean(mds_terms,k=4)

temp = fread("./training_data/model_input/test_y.csv")
