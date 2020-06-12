#install.packages("devtools")
library(ggplot2)

devtools::install_github("almogsi/TATE")

library(TATE)

temp5 = NoVAD("I'm sad")

library(dimRed)
data_set <- loadDataSet("3D S Curve", n = 1000)
embed_methods <- c("Isomap", "PCA","MDS","LLE","KPCA","nMDS","AutoEncoder","tSNE","UMAP")
data_emb <- lapply(embed_methods, function(x) embed(data_set, x))
names(data_emb) <- embed_methods
plot_R_NX(data_emb)


curve_df = as.data.frame(data_set)
