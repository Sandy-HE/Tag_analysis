#Date: 13-05-2020

library(text2vec)  
library(data.table) 
library(stringr)
library(tibble)
library(stopwords)
library(dplyr)
library(ggplot2)


##initial
#alltags <- fread("alltranstags_phrase.csv")
train <- fread("alltranstags_phrase.csv")

##prepare train set and test set
##divide into train set and test set with the proportion of 4:1
#all_ids <- alltags$track_id 
#train_ids <- sample(all_ids, nrow(alltags)/5*4)
#test_ids <- setdiff(all_ids, train_ids) 
#train = alltags[alltags$track_id %in% train_ids,]  
#test = alltags[alltags$track_id %in% test_ids,]  

#==== text2vec: generate term-co-occurrence matrix ====
##use word_tokenizer for single word analysis
#tok_fun <- word_tokenizer
#it_train <- itoken(train$transtags,   
#                  preprocessor = prep_fun,   
#                  tokenizer = tok_fun,   
#                  ids = train$track_id,   
#                  progressbar = T)  

##use space tokenizer for pharase analysis by separating items through ";"
tokens <- space_tokenizer(train$transtags, sep=";")
it_train <- itoken(tokens, 
                   ids = train$track_id,   
                   progressbar = T) 

##set stop words and remove them from vocabulary
#stop_words <- c("the","to","of","and","a","in","this","-","for","it","that","on","is")
stop_words<- stopwords(language = "en", source = "snowball")
stop_words<-stop_words[-81:-98]
stop_words<-stop_words[-147:-149]
#stop_words2<- stopwords(language = "en", source = "smart")
system.time(vocab <- create_vocabulary(it_train ,stopwords = stop_words))

#=========only keep emotion corpus====
#"emotion_terms_list_statistic.csv" is a collection of emotion terms from multiple sources
emo_terms = fread("emotion_terms_list_statistic.csv")
pruned_vocab= vocab[vocab$term %in% emo_terms$term,]
pruned_vocab = prune_vocabulary(pruned_vocab,   
                                doc_count_min= 100)  
#In out trial, 106 emotion terms are left
#=========End of only keep emotion corpus====

#=====Remove some non-related words====
term_ex = fread("non-related_terms.csv")
term_ex_list = c(term_ex$judge,term_ex$ambigu,term_ex$redun,term_ex$phrase)
pruned_vocab= vocab[!vocab$term %in% term_ex_list,]
pruned_vocab = prune_vocabulary(pruned_vocab,   
                                term_count_min = 1000,   
                                doc_proportion_max = 0.8,  
                                doc_proportion_min = 0.0002)  

#=====End of Removing some non-related words====
##remove low-freq and high-freq words
pruned_vocab = prune_vocabulary(vocab,   
                                term_count_min = 1000,   
                                doc_proportion_max = 0.8,  
                                doc_proportion_min = 0.0002)  

##set up corpus vector
vectorizer <- vocab_vectorizer(pruned_vocab)

##generate DTM(doc-term matrix). Here we can understand it as song-term matrix
system.time(dtm_train <- create_dtm(it_train, vectorizer))


#==== generate TCM(term-co-occurrence matrix). ====
system.time(tcm_train <- create_tcm(it_train, vectorizer, skip_grams_window = 5L,
                                    skip_grams_window_context = c("right")))
#skip_grams_window_context = c("symmetric")

#sometimes, the matrix induce big cost when training. 
#But in my experiment, normalization impact the training result badly
temp = as.matrix(tcm_train)
temp1 = log(tcm_train+1)
dtm_train_l1_norm = normalize(dtm_train, "l1")
temp2 = normalize(tcm_train, "l2")
temp1 = normalize(dtm_train, "l1")
#====LSA model====
#In LSA model, result "lsa$components" is the topic-term matrix. 
#We can transpose it and regard it as word vectors
tfidf = TfIdf$new()
lsa = LSA$new(n_topics = 4)
track_embedding =  fit_transform(dtm_train, tfidf)
track_embedding2 =  fit_transform(track_embedding, lsa)
# track_embedding = dtm_train %>%
#  fit_transform(tfidf) %>%
#  fit_transform(lsa)
temp <- lsa$components
word_vectors <- t(temp)
eterms_df <- fread("./emotion_terms_list2.csv")
sub_wordvec <- subset(word_vectors, rownames(word_vectors) %in% eterms_df$term)
sub_wordvec <- sub_wordvec[order(rownames(sub_wordvec)),]
wordvec_df <- as.data.frame(word_vectors)
wordvec_df$term <- pruned_vocab$term
#fwrite(as.data.frame(word_vectors), "wordvec_<size>.csv")
fwrite(wordvec_df, "phrase_emotion_only_LSA_D4.csv")
#====The end of LSA model====

#====LDA model====
#In LDA model, when setting topic number is 10, it looks good in 4 quadrants.
#But when setting topic number is 20 or 50, it becomes worse and worse. 
#Most topics tend to cluster to one position

#lda_model = LDA$new(n_topics = 30, doc_topic_prior = 0.1, topic_word_prior = 0.01)
#track_topic_distr = 
#  lda_model$fit_transform(x = dtm_train, n_iter = 1000, 
#                          convergence_tol = 0.001, n_check_convergence = 25, 
#                          progressbar = FALSE)
#library(LDAvis)
#lda_model$plot()

#temp <- lda_model$get_top_words(n = 30, topic_number = c(1L, 5L, 10L), lambda = 0.6)
#temp <- lda_model$get_top_words(n = 10, lambda = 0.6)
#topic50_top30 <- as.data.frame(temp, stringsAsFactors = FALSE)


#==These steps saves visualization to dir
#DIR = "LDAvis_topic50"
#lda_model$plot(out.dir = DIR)
#temp1<-lda_model$topic_word_distribution
#topic50_term_matrix <- as.data.frame(temp1, stringsAsFactors = FALSE)
#====The end of LDA model====

#====Glove model===
#This model will use tcm as input, output is word vectors
#You can adjust vector size
#glove = GlobalVectors$new(word_vectors_size = 100, vocabulary = pruned_vocab, x_max = 10)
#wv_main = glove$fit_transform(tcm_train, n_iter = 25, convergence_tol = 0.005)
#wv_context = glove$components
#word_vectors = wv_main + t(wv_context)learning_rate=0.15
glove = GlobalVectors$new(rank = 64, x_max = 10)
wv_main = glove$fit_transform(tcm_train, n_iter = 25, convergence_tol = 0.001, n_threads = 2)
wv_context = glove$components
word_vectors = wv_main + t(wv_context)


#Save word vectors if neccesary, because when you change vector size, the word vector is changed time by time. 
#If you want to get stable result. Just save your result.
wordvec_df <- as.data.frame(word_vectors)
wordvec_df$term <- pruned_vocab$term
#fwrite(as.data.frame(word_vectors), "wordvec_<size>.csv")
fwrite(wordvec_df, "phrase_glove_1756_l2_D300.csv")

#read any back up data
wordvec_df <- fread("phrase_7685_right_glove_D64.csv")
word_vectors <- as.matrix(wordvec_df[,1:64])
rownames(word_vectors) <- wordvec_df$term

#one test case for checking validation of word vector 
library(wordcloud)
temp <- word_vectors["sad", , drop = FALSE]
cos_sim = sim2(x = word_vectors, y = temp, method = "cosine", norm = "l2")
#head(sort(cos_sim[,1], decreasing = TRUE),50)
temp1= sort(cos_sim[,1], decreasing = TRUE)
temp3 = as.data.frame (temp1)
colnames(temp3)=c("freq")
temp3$word = rownames(temp3)

set.seed(1234) # for reproducibility 
wordcloud(words = temp3$word, freq = temp3$freq, min.freq = 0.1,
          max.words=50,random.color=TRUE, random.order=FALSE, rot.per=0.1,
          colors=brewer.pal(8, "Dark2"))
library(wordcloud2)
temp4 = temp3[,c("word","freq")]
wordcloud2(temp4, size=1.6, color=brewer.pal(8, "Dark2"))
#====The end of Glove model====

#====calculate word-word similarity====
#1.get subset of word vectors based on emotion terms list
eterms_df <- fread("./emotion_terms_list.csv")

#option1: if use glove processing result directly
wordvec_df <- as.data.frame(word_vectors)
wordvec_df$term <- rownames(wordvec_df)
#option2: read repository data
wordvec_df <- as.data.frame(fread("wordvec_7685_new.csv"))

#====merge some same meaning terms====
##Solution1: merge terms, then get sub word vectors
termassimilation <- function(item, df){
  termkey <- item[1]
  termset <- item[2]
  df$term[str_detect(df$term, termset)] <- termkey
  df <- aggregate(df[,-101], by=list(term=df$term), mean)
  df[df$term %in% termkey,]
}

system.time(result <- apply(eterms_df,1,termassimilation, df=wordvec_df))  #a list of data frames
a <- bind_rows(result)
rownames(a) <- a$term
a <- a[,-1]
sub_wordvec <- as.matrix(a)

##Save merged result for future use ()
a$term <- rownames(a)
fwrite(a, file="7685_merge_result_101terms_new.csv")

##Get the ready-use data
a<-as.data.frame(fread("7685_merge_result_101terms.csv"))
rownames(a) <- a$term
##please check which column is term, then specify the column number
a <- a[,-1]
sub_wordvec <- as.matrix(a)

##Solution2: not merge terms, get sub word vectors
rownames(wordvec_df)<-wordvec_df$term
word_vectors <- as.matrix(wordvec_df[,-101])
sub_wordvec <- subset(word_vectors, rownames(word_vectors) %in% eterms_df$term)
sub_wordvec <- sub_wordvec[order(rownames(sub_wordvec)),]

#2.calculate pairwise-rows cosine similarity and generate a similarity matrix
termsim_mt <-lsa::cosine(t(sub_wordvec))
#====The end of calculating similarity====

#====MDS model====
##In MDS model, we use dissimilarity instead of similarity.
##When I use similarity in the begining, the model result is 0or1 dim. It is meaningless.

##get dissimilarity
termdis_mt <- max(termsim_mt)-termsim_mt
##MDS model, K is the dimension of target matrix
system.time(mds_terms<-cmdscale(termdis_mt, k=2))


##Visualization
mds_terms.names = rownames(sub_wordvec)
plot(mds_terms[,1],mds_terms[,2],type='p',pch=16,col="grey")
text(mds_terms[,1],mds_terms[,2],mds_terms.names,adj=c(0,1),cex=.7)

#====MDS model 2====
library(vegan)
mds_model<-metaMDS(termdis_mt, k=2,stress=2)
mds_terms <- mds_model$points
plot(mds_terms, type= 'n')
text(mds_terms,mds_terms.names,cex=.7)

#====procrustes analysis====
library(vegan)
library("plotrix")

#====scale classical coordinate data====
#If we do not run this step, the plots in rotated Y is very closed to each other.
#To let rotatedY be sparse in the [-1,1] circle, we can scale X
#And use scaledX as targetX to do procruste analysis with Y, 
#then the rotated Y is sparsed better.
scherer_cord = as.data.frame(fread("scherer_emotion_coord.csv"))
cord_min = min(min(scherer_cord$x),min(scherer_cord$y))
cord_max = max(max(scherer_cord$x),max(scherer_cord$y))
scherer_cord$norm1_x = (scherer_cord$x)/230
scherer_cord$norm1_y = (scherer_cord$y)/230
rownames(scherer_cord) <- scherer_cord$term

scherer_cord_new <- scherer_cord[(rownames(scherer_cord) %in% mds_terms.names),]

mdsterm_df <- as.data.frame(mds_terms)
mdsdata <- mdsterm_df
mdsdata$term <- rownames(mdsterm_df)
fwrite(mdsdata,"./mdsdata.csv")
scherer_cord_new1 <- scale(scherer_cord_new[,-c(1,2,3)])
scherer_cord_new2 <- as.matrix(scherer_cord_new[,-c(1,2,3)])
#rownames(mdsterm_df) <- mds_terms.names
target <- mdsterm_df[rownames(scherer_cord_new),]
target_ex <- mdsterm_df[setdiff(mds_terms.names,rownames(scherer_cord_new)),]

#fwrite(target, file = "Y.csv")
#fwrite(target_ex, file = "other.csv")
#fwrite(scherer_cord_new[,-c(1,2,3)], file = "X.csv")

#X is the classical model, Y is my model
proc <- procrustes(scherer_cord_new1, target)
summary(proc)
max(proc$Yrot)
temp <- proc$Yrot
proc$Yrot <- scale(temp,scale = FALSE)
## S3 method for class 'procrustes'
plot(proc,kind = 0)
plot(c(-1.1,1.1), c(-1.1,1.1), type='n', asp=1,main = "Emotion Dimensional Model")
draw.circle(0, 0, 1, nv = 1000, border = NULL, col = NA, lty = 1, lwd = 1)
arrows(c(-1.1,0),c(0,-1.1),c(1.1,0),c(0,1.1), length=0.1)

points(proc$Yrot,pch=16, col="blue")
text(proc$Yrot,adj=c(0,1),rownames(proc$Yrot), cex = .7)
newdata <- predict(proc,target_ex)
points(newdata, pch=8, col="green")
text(newdata,adj=c(0,1),rownames(newdata), cex=.7)
points(scherer_cord_new2,pch=1, col="red")
arrows(proc$Yrot[,1],proc$Yrot[,2], scherer_cord_new2[,1],scherer_cord_new2[,2],length=0.1, color="blue")
text(scherer_cord_new[,c(4,5)],adj=c(0,1), rownames(proc$Yrot), cex=.7, color= "red")

text(x=0.22,y=1.1, "Arousal", font=2)
text(x=1.2,y=0.1, "Valence", font=2)
text(x=1.15,y=-0.05, "positive" , cex=.7, color="grey", font=3)
text(x=-1.15,y=-0.05, "negative" , cex=.7, color="grey", font=3)
text(x=-0.15,y=1.05, "active" , cex=.7, color="grey", font=3)
text(x=-0.15,y=-1.05, "deactive" , cex=.7, color="grey", font=3)


#==== T-SNE====
library(Rtsne)

tsne <- Rtsne(word_vectors, dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)
#plot(tsne$Y, t='n', main="tsne")
#text(tsne$Y, labels=rownames(word_vectors))

tsnedf <- as.data.frame(tsne$Y)
#fwrite(tsnedf, "tsne_2d_3064_coord.csv")
tsnedf$term <- rownames(word_vectors)


#sample  <- tsnedf[c("happy","sad songs","angry","relax","calm","sleepy"),]
ggplot(a, aes(x=V1, y=V2)) +  
  geom_point(size=1) +
  #guides(colour=guide_legend(override.aes=list(size=6))) +
  xlab("") + ylab("") +
  ggtitle("t-SNE") +
  #theme_light(base_size=20) +
  #theme(axis.text.x=element_blank(),
  #      axis.text.y=element_blank()) +
  geom_text(label=a$term, angle=45)
  #scale_colour_brewer(palette = "Set2")

#More visualization tips for tsne result
#https://www.r-bloggers.com/playing-with-dimensions-from-clustering-pca-t-sne-to-carl-sagan/
## keeping original data
#tsnedf_clusters=tsnedf

## Creating k-means clustering model, and assigning the result to the data used to create the tsne
#fit_cluster_kmeans=kmeans(scale(tsnedf), 4)  
#tsnedf_clusters$cl_kmeans = factor(fit_cluster_kmeans$cluster)

## Creating hierarchical cluster model, and assigning the result to the data used to create the tsne
#fit_cluster_hierarchical=hclust(dist(scale(tsnedf)))

## setting 4 clusters as output
#tsnedf_clusters$cl_hierarchical = factor(cutree(fit_cluster_hierarchical, k=4))  

#plot_cluster=function(data, var_cluster, palette)  
#{
#  ggplot(data, aes_string(x="V1", y="V2", color=var_cluster)) +
#    geom_point(size=0.25) +
#    guides(colour=guide_legend(override.aes=list(size=6))) +
#    xlab("") + ylab("") +
#    ggtitle("") +
#    theme_light(base_size=20) +
#    theme(axis.text.x=element_blank(),
#          axis.text.y=element_blank(),
#          legend.direction = "horizontal", 
#          legend.position = "bottom",
#          legend.box = "horizontal") + 
#    scale_colour_brewer(palette = palette) 
#}

#plot_k=plot_cluster(tsnedf_clusters, "cl_kmeans", "Accent")  
#plot_h=plot_cluster(tsnedf_clusters, "cl_hierarchical", "Set1")







#check the order of documents in both dtm and original dataset
identical(rownames(dtm_train), train$track_id) 

topwords <- c("happy","sad","angry","relaxing","upbeat",
              "melancholy","fun","chill","party","at ease",
              "sweet","mellow","joyful","brutal","dancing",
              "calm","sleepy","depressive","black","passionate",
              "moody","quirky","sexy","dark","soulful")
topwords <- c("happy","sad","angry","relaxed","upbeat",
              "melancholy","fun","chillout","party","easy listening",
              "sweet","mellow","brutal","dancing")
##prune matrix only containing topN-freq tags,'topwords' dataset determines which phrases are kept
tcm_sample <- tcm_train[rownames(tcm_train) %in% topwords,colnames(tcm_train) %in% topwords]
tcm_sample_mat <- as.matrix(tcm_train)
tcm_sample_mat <- as.matrix(tcm_sample)
##for whole train set
#tcm_sample_mat <- as.matrix(tcm_train)

##clean and normalize the term-cooccurrence matrix
tcm_sample_diag <- diag(diag(tcm_sample_mat))
tcm_sample_whole <- tcm_sample_mat + t(tcm_sample_mat) - tcm_sample_diag*2
#tcm_sample_whole <- tcm_sample_mat - tcm_sample_diag
tcm_sample_whole <- round(tcm_sample_whole,2)   # this is matrix
#tcm_sample_whole <- as.matrix(tcm_sample_whole)
words <- colnames(tcm_sample_whole)
words <- order(words)
##Please compare the weight before and after ordering, sometimes it does not work
tcm_sample_whole1 <- tcm_sample_whole
tcm_sample_whole <- tcm_sample_whole[words,words]
#tcm_sample_whole <- tcm_sample_whole[,words]

##shrink the whole weight scale
#tcm_sample_whole_log <- log(tcm_sample_whole+1)
#tcm_sample_whole <- round(tcm_sample_whole_log,2)

##reduce edges for simplified network visualization
#tcm_sample_whole_interm <- ifelse(tcm_sample_whole==0, NA, tcm_sample_whole)
#md <- median(tcm_sample_whole_interm, na.rm = T)
#tcm_sample_less_edges <- ifelse(tcm_sample_whole<md, 0, tcm_sample_whole)

##data backup
#tcm_sample_df <- as.data.frame(tcm_sample_whole, stringsAsFactors = F)
#tags <- rownames(tcm_sample_df)
#tcm_sample_df<- cbind(tags,tcm_sample_df)
#fwrite(tcm_sample_df, file = "tcm_sample_pruned_vocab.csv"

#==== visualization ====
#set.seed(10)
library(igraph)
##build up nodes and edges for the whole sample data or less-edges data
topwordsnode <- as.data.frame(rownames(tcm_sample_whole),stringsAsFactors = FALSE)
#topwordsnode <- as.data.frame(rownames(tcm_sample_less_edges),stringsAsFactors = FALSE)
colnames(topwordsnode) <- "tags"
topwordsnode$id <- rownames(topwordsnode)
topwordsedge <- graph.adjacency(tcm_sample_whole,weighted=TRUE)
#topwordsedge <- graph.adjacency(tcm_sample_less_edges,weighted=TRUE)
topwordsedge <- get.data.frame(topwordsedge)

##backup data
fwrite(topwordsnode, file="items_node.csv")
fwrite(topwordsedge, file="items_edge.csv")



#==== network package usage ====
library(network)
library(tnet)

##build up tagsnetwork
tagsnetwork <- network(topwordsedge, vertex.attr = topwordsnode, matrix.type="edgelist",loops=F, multiple=F, ignore.eval = F)

##define some attributes for edge and vertex
network::set.edge.attribute(tagsnetwork, "weight", topwordsedge$weight)
t <- as.edgelist(tagsnetwork, attrname = "weight")%>%
  as.tnet %>%
  degree_w

network::set.vertex.attribute(tagsnetwork, "degree_w", t[, "output" ])

#l <- tagsnetwork %v% "degree_w"

l <- network.vertex.names(tagsnetwork)
network::set.vertex.attribute(tagsnetwork, "label", l)
#tagsnetwork <- network(tcm_sample_whole, matrix.type="adjacency")
#plot(tagsnetwork, vertex.cex = 3,mode = "circle")
#==== End of network package usage ====



#==== ndtv: dynamic network visualization ====
library("ndtv")

#reuse tagsnetwork

len <- ncol(tcm_sample_whole)
times <- nrow(topwordsedge)
vs <- data.frame(onset=0, terminus=times+1, vertex.id=1:len)
es <- data.frame(onset=1:times, terminus=times+1, 
                 head=as.matrix(tagsnetwork, matrix.type="edgelist")[,1],
                 tail=as.matrix(tagsnetwork, matrix.type="edgelist")[,2])

tagsnetwork.dyn <- networkDynamic(base.net=tagsnetwork, edge.spells=es, vertex.spells=vs)
compute.animation(tagsnetwork.dyn, animation.mode = "kamadakawai",
                  slice.par=list(start=0, end=times, interval=1, 
                                 aggregate.dur=1, rule='any'))

render.d3movie(tagsnetwork.dyn, usearrows = F, displaylabels = T, bg="#ffffff", 
               vertex.border="#111111", vertex.col =  "#aaaaaa",
               vertex.cex = log(tagsnetwork %v% "degree_w")/10, 
               label.cex = 0.5,
               edge.lwd = log(tagsnetwork %e% "weight")/5, edge.col = '#55555599',
               vertex.tooltip = tagsnetwork %v% 'tags',
               edge.tooltip =tagsnetwork %e% "weight",
               launchBrowser=F, filename="Media-Network.html",
               #render.par=list(tween.frames = 30, show.time = F),
               plot.par=list(mar=c(0,0,0,0),xlim=c(-1.5,1.5),ylim=c(-1.5,1.5))
)  


#==== cosine similarity====
library('lsa')

act_baseline <- read.csv("ACTemotion_coord_new.csv",stringsAsFactors = F )

temp <- combn(act_baseline$item,2)

cos_simi <- function(x) lsa::cosine(as.numeric(as.vector(act_baseline[act_baseline$item==x[1],2:3])),as.numeric(as.vector(act_baseline[act_baseline$item==x[2],2:3])))

simi_list <- apply(temp,2, cos_simi)
pair_list <- apply(temp,2, function(x) paste(x[1],x[2], sep = "-"))
base_df <- cbind(pair_list,round(simi_list,2))



cos_simi_terms <- function(x){
  #tcm_new <- tcm_sample_whole[x, -which(colnames(tcm_sample_whole) %in% x)]
  tcm_new <- tcm_sample_whole[x,]
  lsa::cosine(tcm_new[1,],tcm_new[2,])
}

simi_term_list <- apply(temp,2,cos_simi_terms)
final_df <- cbind(base_df, round(simi_term_list,2))
final_df <- as.data.frame(final_df, stringsAsFactors = F)
colnames(final_df) <- c("pairterms","basesim","termssim")
final_df$basesim <- as.numeric(final_df$basesim)
final_df$termssim <- as.numeric(final_df$termssim)
#normalized into [-1,1]
bottomline <- min(final_df$termssim)
topline <- max(final_df$termssim)
final_df$norm_termssim <- round((final_df$termssim-bottomline)/(topline-bottomline)*2-1,2)

library(ggplot2)
ggplot(final_df,aes(basesim,norm_termssim))+
geom_smooth(method="lm")+
 geom_point()
  
final_simi <- lsa::cosine(final_df$basesim,final_df$termssim)

#====test similarity from last.fm file====
#get 
#semat <- as.matrix(dtm_train)*
songslist <- c("TRAAAAK128F9318786","TRZNRZF128F9318787","TRRJFIC128F931879D","TRLYTDK128F93187BB","TRAOCLB128F92C2696")

alltags <- fread("./statwords.csv")
subtags <- alltags[alltags$track_id%in% songslist,]

itemsMultiTrans <- function(items){
  itemlist <- tstrsplit(items, "\\*")
  itemdf <- as.data.frame(itemlist, stringsAsFactors = F,col.names = c("item","popularity"))
  #if(!is.integer(itemdf$popularity)) print(itemdf)
  #for single word
  #itemdf$multiItem <- strrep(paste0(itemdf$item," "),as.integer(itemdf$popularity) )
  #for phrase
  #itemdf$multiItem <- strrep(paste0(itemdf$item,";"),as.integer(itemdf$popularity) )
  #itemdf <- itemdf[itemdf$popularity!=0,]
  
  #itemsstr<-paste(itemdf$multiItem,collapse =" ")
  #itemsstr
  print(itemdf$item)
  subtags[itemdf$item]<-itemdf$popularity
  subtags
}

obsv <- str_to_lower(subtags$tags)
obsv <- str_split(obsv,pattern = ";")      #list
obsv1 <- sapply(obsv, itemsMultiTrans)    #char vector

dtm_sample <- dtm_train[rownames(dtm_train) %in% songslist,]
dtm_sample_mat <- as.matrix(dtm_sample)
temp2 <- lsa::cosine(dtm_sample_mat[1,],dtm_sample_mat[2,])
