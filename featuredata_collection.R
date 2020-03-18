library("rhdf5")
library(ggplot2)
library("data.table")
library(reticulate)
np <- import("numpy")



songsData <- fread("allsongsdata_300seg.csv")
all_songs_withtags = dtm_train[rownames(dtm_train) %in% songsData$track_id,]
all_songs_withtags1 = as.matrix(all_songs_withtags)


#tcm_new is terms*emotion_term co-occurrence matrix
temp2=as.matrix(tcm_train)
eterms_df <- read.csv("./emotion_terms_list_40.csv")
eterms = eterms_df$term
#diag check
diag_check = temp2[rownames(temp2) %in% eterms,colnames(temp2) %in% eterms]

temp3 = temp2 + t(temp2)- diag(diag(temp2))
#which(temp3!=0,arr.ind = T)

temp4 = temp3/max(temp3)*30000
#+diag(1,nrow(temp2),nrow(temp2))
#temp5 = atan(temp4)*2/pi
temp5 = tanh(temp4)
temp5 = round(temp5,2)
#temp5 = temp5-diag(diag(temp5))+diag(1,nrow(temp5),nrow(temp5))

tcm_new = temp5[,colnames(temp5) %in% eterms]
#round(tcm_new['sad',],2)


temp5 = all_songs_withtags%*%tcm_new
#temp6 = log(temp5+1)
temp6 = temp5/max(temp5)*50
temp6 = atan(temp6)*2/pi
temp6 = round(temp6,2)
#temp7 = temp6[ rownames(temp6) %in% c("TRBAADF128F423CCD2"),]
#temp7


#get the index of songs data with tags info
temp4 = matrix(1:nrow(songsData), ncol = 1)
rownames(temp4) = songsData$track_id
colnames(temp4) = c("index")

index = as.vector(temp4[rownames(all_songs_withtags),])



pitchData=getPitchData()
pitchData = pitchData[,,index]

timbreData = getTimbreData()
timbreData = timbreData[,,index]

loudnessData = getLoudnessData()
loudnessData = loudnessData[index,]

loudnessStart = getLoudnessStart()
loudnessStart=loudnessStart[index,]

loudnessMaxtime=getLoudnessMaxtime()
loudnessMaxtime=loudnessMaxtime[index,]

loudnessAll = loudnessAll[,,]



library(reticulate)
np <- import("numpy")
np$savez("A_pitchdata_300seg_withtags.npz", pitchData)
np$savez("A_timbredata_300seg_withtags.npz", timbreData)
np$savez("A_loudnessdata_300seg_withtags.npz", loudnessAll)

#np$savez("alltags_300seg_Y.npz",temp6)    // can not be run in my PC(maybe admin issue)
save(as.matrix(temp6), file="alltags_300seg_Y.RData")


# if(sum(is.na(loudnessAll))>0) {
#   print("true")
# }



