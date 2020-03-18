library(abind)



load("./2019msdanalysis/Z_loudnessdata_300seg_withtags.RData")

#temp3 <- loudnessAll

temp3 <- abind(temp3, loudnessAll,along=1)


save(temp, file="AtoI_loudnessdata_300seg_withtags.RData")
save(temp1, file="JtoO_loudnessdata_300seg_withtags.RData")
save(temp2, file="PtoU_loudnessdata_300seg_withtags.RData")
save(temp3, file="VtoZ_loudnessdata_300seg_withtags.RData")
