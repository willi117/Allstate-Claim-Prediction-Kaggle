require(h2o)
require(doMC)

#parellel processing in mac2
registerDoMC()#mac parrallel proceessing 
options(cores=4)#5 cores 
getDoParWorkers()#check
foreach(i=1:3) %dopar% sqrt(i)

h2o.init(nthreads=-1, max_mem_size="2G")

setwd(Users/joshuawilliams/Documents/Multi-Variate Analysis/Final/Final)
train.tune.hex.or <- h2o.uploadFile("train.csv")
train.tune.hex <- train.tune.hex.or[1:131822,]
validation.tune.hex <- train.tune.hex.or[131823:188318,]


hidden_opt=list(c(1024,1024,2048), c(1024,1024), c(1024,500,1024))
l1_opt =seq(1e-6,1e-3,1e-5)

hyper_params <- list(hidden=hidden_opt,l1=l1_opt)
search_criteria=list(strategy = "RandomDiscrete", max_models=15, max_runtime_secs=3600, seed=54096)
y<-"loss"
x<-setdiff(names(train.tune.hex.or),y)

#train.tune.hex[,y] <- as.factor(train.tune.hex[,y])
#validation.tune.hex[,y] <- as.factor(validation.tune.hex[,y])


#Technically this random grid search is wrong. It is running it as if its regression intead of classification.
#The issue is that the part of the data is classification and the remaining columns are continuous data. 



randomGridSearch <- h2o.grid(x=x, y=y, 
                             algorithm = "deeplearning",
                             activation="Rectifier",
                             grid_id = "RGS2", 
                             training_frame= train.tune.hex,
                             validation_frame= validation.tune.hex ,
                             score_interval=2,
                             epochs=100,
                             hyper_params=hyper_params,
                             search_criteria = search_criteria,
                             stopping_rounds=3,
                             stopping_tolerance=.05,
                             stopping_metric="AUTO"
)
h2o.getGrid('RGS2') #1189.697 MAE

# pca
traino <- read.csv('train.csv')
for (i in 2:117) {traino[,i] <- as.numeric(traino[,i])}
validation <- traino[131823:188318,]
train <- traino[1:131822,]
loss <- train$loss 
train$loss<- NULL
Pca <- prcomp(train[,-1], scale. = T, center = T)
trainPca <- data.frame(id=train$id,Pca$x[,1:68],loss= loss)#85% of varience
loss1 <- validation$loss
validation$loss <- NULL
Pca2 <- predict(Pca,validation[,-1])
validationPca <- data.frame(id=validation$id,Pca2[,1:68],loss=loss1)
write.csv(trainPca,'trainPca.csv',row.names = F)
write.csv(validationPca, 'validationPCA.csv',row.names = F)




require(h2o)



h2o.init(nthreads=-1, max_mem_size="2G")




trainPca.tune.hex <- h2o.uploadFile('trainPca.csv')
validationPca.tune.hex <- h2o.uploadFile('validationPCA.csv')


#now run define hyperperameters same as before




hidden_opt=list(c(1024,1024,2048), c(1024,1024), c(1024,500,1024))
l1_opt = seq(1e-6,1e-3,1e-5)
activation_opt = c('Rectifier', "Tanh")

hyper_params <- list(hidden=hidden_opt,l1=l1_opt, )
search_criteria=list(strategy = "RandomDiscrete", max_models=15, max_runtime_secs=3600)
y <- "loss"
x <- setdiff(names(trainPca.tune.hex),y)


#h20 grid search model
randomGridSearch <- h2o.grid(x=x, y=y, 
                             algorithm = "deeplearning",
                             activation="Rectifier",
                             grid_id = "RGS4", 
                             training_frame= trainPca.tune.hex,
                             validation_frame= validationPca.tune.hex,
                             score_interval=2,
                             epochs=100,
                             hyper_params=hyper_params,
                             search_criteria = search_criteria,
                             stopping_rounds=3,
                             stopping_tolerance=.05,
                             stopping_metric="AUTO"
)


summary(radomGridSearch)





# real model 1231.66229


#pca full train

traino <- read.csv('train.csv')
test <- read.csv('test.csv')
for (i in 2:117) {traino[,i] <- as.numeric(traino[,i])}
loss <- traino$loss 
traino$loss<- NULL
Pca <- prcomp(traino[,-1], scale. = T, center = T)
trainPca <- data.frame(id=traino$id,Pca$x[,1:68],loss= loss)#85% of varience



loss1 <- test$loss
test$loss <- NULL
for (i in 2:117) {test[,i] <- as.numeric(test[,i])}
Pca2 <- predict(Pca,test[,-1])
testPca <- data.frame(id=test$id,Pca2[,1:68])

write.csv(trainPca,'trainPcaWhole.csv',row.names = F)
write.csv(testPca,'testPcaWhole.csv',row.names = F)

require(h2o)



h2o.init(nthreads=-1, max_mem_size="2G")


setwd(Users/joshuawilliams/Documents/Multi-Variate Analysis/Final/Final)

train.hex <- h2o.uploadFile('trainPcaWhole.csv') #upload data to h2o
test.hex<- h2o.uploadFile('testPcaWhole.csv') #upload train to h2o



y <- "loss"
x <- setdiff(names(train.hex),y)


mydeep = h2o.deeplearning(x=x, y=y , 
                          overwrite_with_best_model=T,
                          training_frame=train.hex,
                          nfolds=10,
                          single_node_mode = T,
                          activation="Rectifier",
                          hidden=c(512,512),
                          epochs=100,
                          train_samples_per_iteration=-2,
                          seed=600,
                          l1=5.7e-4,
                          adaptive_rate=T,
                          nesterov_accelerated_gradient=TRUE,
                          stopping_metric="AUTO",
)





pred<-h2o.predict(mydeep, test.hex)
predframe <- data.frame(test$id, pred)
View(pred)
pred2 <- predict(mydeep, test.hex)
write.csv(as.data.frame(pred2), "predict.csv")
sub<-read.csv("predict.csv")
colnames(sub) <- c("id", "loss")
write.csv(sub, "sub1PcaNN.csv", row.names = F)




#try without pca

install.packages(h20)
require(h2o)

h2o.init(nthreads=-1, max_mem_size="2G")


setwd(Users/joshuawilliams/Documents/Multi-Variate Analysis/Final/Final)

train.hex <- h2o.uploadFile('train.csv') #upload data to h2o
test.hex<- h2o.uploadFile('test.csv') #upload train to h2o



y <- "loss"
x <- setdiff(names(train.hex),y)


mydeep = h2o.deeplearning(x=x, y=y , 
                          overwrite_with_best_model=T,
                          training_frame=train.hex,
                          nfolds=10,
                          single_node_mode = T,
                          activation="Rectifier",
                          hidden=c(512,512),
                          epochs=100,
                          train_samples_per_iteration=-2,
                          seed=600,
                          l1=5.7e-4,
                          adaptive_rate=T,
                          nesterov_accelerated_gradient=TRUE,
                          stopping_metric="AUTO",
)





#Support Vector Machine 
require(doMC)
#parellel processing in mac2
registerDoMC()#mac parrallel proceessing 
options(cores=5)#5 cores 
getDoParWorkers()#check


setwd(Users/joshuawilliams/Documents/Multi-Variate Analysis/Final/Final)
test<- read.csv('test.csv')
pred<-h2o.predict(mydeep, test.hex)
View(pred)
pred2 <- predict(mydeep, test.hex)
write.csv(as.data.frame(pred), "predict2.csv")
sub2<-read.csv("predict2.csv")
colnames(sub2) <- c("id", "loss")
sub2$id <- test$id
write.csv(sub2, "sub2NN.csv", row.names = F)

require(snowfall)
sfInit(parallel = T, cpus = 4)
sfStop(nonstop=)
#SVM (took to long stopped at 30 hours)
require(e1071)
train <- read.csv('train.csv')
test <- read.csv('test.csv')
tune <- tune.svm(loss~., data=train[22903:22913,], gamma  = 10^(-6:-1), cost = 10^(1:2))

summary(tune)
modelsvm <- svm(loss~., data= train, gamma= .005, cost = 80) #took to long to run (~30 hours) stopped



#Memory Management
rm(list=ls())  #clears the workspace
memory.size(max=T)  #setting the memory to maximum
library(doParallel)  #





