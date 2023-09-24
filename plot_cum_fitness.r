library(pracma) # to estimate the EMWA (media movil)

ppo1 <- read.table('output\\ppo_fitness.txt',sep="",skip =3)
mc1 <- read.table('output\\mc_fitness.txt',sep="",skip =3)
# remove first col
ppo1$V1 <- NULL
mc1$V1 <- NULL 

# add mean and standard deviation
ppo <- transform(ppo1, sum=rowSums(ppo1), mean=rowMeans(ppo1), std=apply(ppo1,1,sd,na.rm=TRUE))
mc <- transform(mc1, sum=rowSums(mc1), mean=rowMeans(mc1), std=apply(mc1,1,sd,na.rm=TRUE))

x <- seq(1,nrow(mc),1)


ppo_chart <- transform(ppo, emwa=movavg(ppo$sum,n=20,type="e"))
mc_chart <- transform(mc, emwa=movavg(mc$sum,n=20,type="e"))

data <- data.frame(movavg(ppo$sum,n=20,type="e"),movavg(mc$sum,n=20,type="e"))
#dev.off()
matplot(data, type="l",pch=1,col=1:2)
