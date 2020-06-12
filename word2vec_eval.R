#Date: 2020-05-18

#To compare glove performance with word2vec, meanwhile keep the same corpus.
#We use text2vec pruned_vocab to provide the input of word2vec
#read back the processing result of word2vec to do evaluation
#word2vec part run by "wvbyword2vec.py"

library("data.table")

#get pruned_vocab from text2vec analysis(combined with lastfm_tags_analysis.R)
#based pruned_vocab, clean tags dataset and save csv file
termfilter <- function(termlist){
  term = termlist[termlist %in% pruned_vocab$term]
  paste(term,collapse =";")
}

token1= lapply(tokens, termfilter)
#temp2 = as.data.frame(do.call(rbind,token1))


temp1 = train[,"track_id"]
temp1$prunedtag= token1
fwrite(temp1, file="allprunedtags_emo_only.csv")


#====read back the result of word2vec processing
#'phrase_emo_only_cbow_D64.csv'
wordvec_df <- fread("phrase_7685_cbow_D64.csv")
wordvec_df <- fread("phrase_7685_sg_D32.csv")
wordvec_df <- fread("phrase_emo_only_cbow_D4.csv")
wordvec_df <- fread("phrase_emo_only_sg_D32.csv")
d=32
wordvec_df <- wordvec_df[-1,]
colnames(wordvec_df)[d+1]<-"term"
word_vectors <- as.matrix(wordvec_df[,1:d])
rownames(word_vectors) <- wordvec_df$term


eterms_df <- fread("./emotion_terms_list3.csv")
sub_wordvec <- subset(word_vectors, rownames(word_vectors) %in% eterms_df$term)
sub_wordvec <- sub_wordvec[order(rownames(sub_wordvec)),]

#====Obtain ground truth data for valence-arousal-dominant rating====
#Warriner VAD rating, value range is [1,9] 
baseline_df = fread("Warriner_avd_ratings.csv")
baseline_df = baseline_df[,c("Word","V.Mean.Sum","A.Mean.Sum","D.Mean.Sum")]
baseline_df = baseline_df[baseline_df$Word %in% rownames(sub_wordvec),]
#Normalized to [-1,1]. Given range [-n,n], scale to [-1,1]. (x-(-n))*2/2n-1
baseline_mt = as.matrix(baseline_df[,2:3])
baseline_mt= (baseline_mt-1)*2/8-1
rownames(baseline_mt) <- baseline_df$Word


#====nMDS -- reduce to 2D or 3D word vectors====
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

#====visualization====
plot(mds_terms, type= 'n')
text(mds_terms,rownames(mds_terms),cex=.7)


#====Scherer baseline====
scherer_cord = as.data.frame(fread("scherer_emotion_coord.csv"))
eterms_df <- fread("./emotion_terms_list.csv")
sub_wordvec <- subset(word_vectors, rownames(word_vectors) %in% eterms_df$term)
sub_wordvec <- sub_wordvec[order(rownames(sub_wordvec)),]

scherer_cord$norm1_x = (scherer_cord$x)/230
scherer_cord$norm1_y = (scherer_cord$y)/230
scherer_cord_new <- scherer_cord[scherer_cord$term %in% rownames(sub_wordvec),]
scherer_cord_new2 <- as.matrix(scherer_cord_new[,-c(1,2,3)])
mds_terms <- mds_terms[rownames(mds_terms) %in% scherer_cord_new$term ,]
proc <- procrustes(scherer_cord_new2,mds_terms)
