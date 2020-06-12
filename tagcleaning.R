#Date: 2020-05-18

#word-level tags set up
# @level1: remove number, all characters except alphabeta and '''. stopwords, negation words and their following words 
# @level2: based on level1, remove some non-English words, some typo error
# @level3: based on level1, remove some non-related-emotion words such as "good", "favorite", "songs"


#library(tidyverse)
library("stringr")
library("data.table")
library(text2vec)
library(stopwords)
library(tm)
library(qdapDictionaries)


train <- fread("alltranstags.csv")

#=====Do level1(lv1) tag cleaning=====
#remove non-alpha&num(most are punctuation) expect "-"
#cleantag = str_replace_all(train$transtags, "[^[:alnum:]-]"," " )
#remove meaningless "-"
#cleantag = str_replace_all(cleantag, "[-]{2,}","")
#cleantag = str_replace_all(cleantag, "\\s+-|-\\s+"," ")
#cleantag = str_replace_all(cleantag, "\\s+-|-\\s+"," ")
#to lowercase
#cleantag = str_to_lower(train$transtags)
#remove numbers(1990, 80s, 20th, giveme5,...)
cleantag = str_replace_all(train2$transtags, "[[:digit:]]", " ")
#remove punctuation except "'"
cleantag = str_replace_all(cleantag, "[^[:alpha:]']"," " )


#remove single letter, warning that it will impact I'm --> ', can't --> can'
#cleantag = str_replace_all(cleantag, "\\b(\\w)\\b", "") 

#remove stop words except negation words
stop_words<- stopwords::stopwords(language = "en", source = "snowball")
neg_words <- c(stop_words[81:98],stop_words[165:167])
supple_neg <- c("neither", "without")
neg_words <- c(neg_words,supple_neg )
stop_words<-stop_words[-81:-98]
stop_words<-stop_words[-147:-149]

cleantag = removeWords(cleantag, stop_words)
#remove redundant white space
cleantag = str_squish(cleantag)
#Before the following action, we do investigation about what kind of negative description are used.
#check the existence of negation word in tag dataset
#we select 23 negation words from stopwords list, and check one by one
train$cleantag = cleantag
corpus_df = train[,c("track_id","cleantag")]
colnames(corpus_df) = c("doc_id", "text")
mycorpus <- VCorpus(DataframeSource(corpus_df))
test = tdm$dimnames$Terms
#check if negation words appear in tag dataset
for(i in neg_words){
  test1 = grep(paste("\\b",i,"\\b",sep=""),test)     #check separate word
  # test1 = grep(i,test)                               #check embeding and separate word
  print(paste(i,": ", test1))
}
#filter out all songs with negation word
findterm <- function(termlist){
  #print(termlist)
  term = grep("neither",termlist,value=T)
  term
}
#temp1 = temp[str_detect(temp$text,"\\b(nor|not|cannot|neither|without)\\b"),]
temp1 = corpus_df[str_detect(corpus_df$text,"\\b(neither)\\b"),]
temp2 = train[train$track_id %in% temp1$doc_id,c("track_id","tags")]
temp3 = str_split(temp2$tags, ";")
temp4 = sapply(temp3, findterm)

#Based on investigation above, we take steps below.
#remove negation words and the first word following them.(here most stop words have been removed)
#for the negation words without words followed, they won't be removed.
#can't stop listening --> listening
#ha ha me neither --> ha ha neither
cleantag = str_replace_all(cleantag,paste(neg_words, "\\b(\\w)+\\b", collapse='|'),"")
#remove single letter, do twice to remove start with single letter and end with single letter
cleantag = str_replace_all(cleantag, "\\b\\w ", " ")
cleantag = str_replace_all(cleantag, " \\w\\b", " ") 
cleantag = str_squish(cleantag)
#convert duplicate characters appeared more than 3 time in one word into one character
#such as thaaaaaank you -> thank you
#cleantag = str_replace_all(cleantag, "(.)[^[eorz]]\\1{2,}","\\1")
#based on conversion, convert words containing eorz.  coooool-> cool, buzzzzzzz-> buzz
#cleantag = str_replace_all(cleantag, "(.)\\1{2,}",str_dup("\\1",2))
train$cleantag = cleantag

#=====Remove some non-related words====
word_ex = fread("non-related_words.csv")
cleantag = cleantag_df1$cleanlv1
#ambiguous words
cleantag = removeWords(cleantag, word_ex$ambigu)
#judge words
cleantag = removeWords(cleantag, word_ex$judge)
#other non-related words
cleantag = removeWords(cleantag, word_ex$redun)

cleantag_df1$cleanlv1=cleantag
fwrite(cleantag_df1, "cleantaglv3.csv")
#=====Do level2(lv2) tag cleaning=====
#remove meaningless words
#create corpus to do further clean
corpus_df = train[,c("track_id","cleantag")]
colnames(corpus_df) = c("doc_id", "text")
mycorpus <- VCorpus(DataframeSource(corpus_df))
tdm <- TermDocumentMatrix(mycorpus )
term_count_min = 10000
all_tokens <- findFreqTerms(tdm,lowfreq = 1000, highfreq = term_count_min)
tokens_to_remove <- setdiff(all_tokens,GradyAugmented)
#when term_count_min(highfreq) is set ot 10K, lowfreq as 1000.  
#Total 6444 meaningless words should be removed. But due to memory limited, 
#we remove in 200 steps and use intermediate variable mycorpus1.
mycorpus1 <- tm_map(mycorpus, content_transformer(removeWords), 
                   tokens_to_remove[6201:6444])
mycorpus <- tm_map(mycorpus1, content_transformer(removeWords), 
                   tokens_to_remove[6001:6200])
#This step is converting corpus format to dataFrame
cleantag_df = data.frame(text=unlist(sapply(mycorpus1, `[`, "content")), 
                   stringsAsFactors=F)

#if only saving cleantag_df(there is only one column), writing operation is OK. 
#But when you read it back, error occurred. One empty line is added at the end of excel file.
#solution: 1. remove empty line manually 2. add track_id as another column and save it.
#fwrite(cleantag_df,file = "cleantag_withpartialremovemeaninglesswords6444.csv")

#=====Backup the tag dataset after cleaning====
tagcorpus <- train[,c("track_id","cleantag")]
colnames(tagcorpus)[2] <- "cleanlv1"
tagcorpus$cleanlv2 <- cleantag_df$text


fwrite(tagcorpus[,c("track_id","cleanlv1")], file="cleantaglv1.csv")
fwrite(tagcorpus[,c("track_id","cleanlv2")], file="cleantaglv2.csv")
#I also tried to save the whole tagcorpus, but it is about 3.8G. Too big. 
#When I tried to read it back, it tooks 6 mins. 
#But if I save them separately as above, it could be read back within 15 sec.


#test = removePunctuation(test, preserve_intra_word_contractions = FALSE,
#                        preserve_intra_word_dashes = TRUE)



