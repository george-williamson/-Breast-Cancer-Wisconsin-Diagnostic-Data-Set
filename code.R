#Xuwen's Part
#input library
rm(list = ls())
library(dplyr)
library(ggplot2)
library(psych)
library(patchwork)
library(readr)
library(corrplot)
library(stringr)
library(xgboost)
library(mlr)
library(data.table)
library(parallel)
library(parallelMap)
library(caret)

#input data
p_theme <- theme(panel.background = element_blank(),
                 panel.grid.major.y = element_line(colour = "grey"),
                 plot.title = element_text(hj>ust = 0.5),
                 legend.position = "top")

wdbc_raw <- read_csv("wdbc.data.csv",col_names = T)

range(wdbc_raw$X1)
dim(wdbc_raw)
#input the colnames
col_names <-read.table(text = "colname
id
label
radius_m
texture_m
perimeter_m
area_m
smoothness_m
compactness_m
concavity_m
concave_m
symmetry_m
fd_m
radius_sd
texture_sd
perimeter_sd
area_sd
smoothness_sd
compactness_sd
concavity_sd
concave_sd
symmetry_sd
fd_sd
radius_w
texture_w
perimeter_w
area_w
smoothness_w
compactness_w
concavity_w
concave_w
symmetry_w
fd_w
",header=T)


#check the missing values
names(wdbc_raw) <- col_names$colname
sum(is.na(wdbc_raw))
sum(!complete.cases(wdbc_raw))
names(wdbc_raw)

wdbc_raw = wdbc_raw[,-33]


# -------------------- XGBoost Implementation -------------------- #
set.seed(22)

# Convert labels from character to integers
lookup <- c("B" = 0, "M" = 1)
wdbc_raw = wdbc_raw
wdbc_raw$label <- lookup[wdbc_raw$label]

# Create a 75/25 Training/Test split
n = nrow(wdbc_raw)
train_indices = sample(n,floor(0.75*n))

train_x = as.matrix(wdbc_raw[train_indices, -c(1,2)])
train_y = wdbc_raw$label[train_indices]

holdout_x = as.matrix(wdbc_raw[-train_indices, -c(1,2)])
holdout_y = wdbc_raw$label[-train_indices]

# Create xgb.DMatrix objects and define parameters
xgb.train = xgb.DMatrix(data=train_x,label=train_y)
xgb.test = xgb.DMatrix(data=holdout_x,label=holdout_y)

# Parameters for 1st Model
# Gamma = 1 to stop overfitting
# Reduced eta to 0.1 to slow down learning
params <- list(
  booster="gbtree",
  eta=0.1,
  max_depth=10,
  gamma=1,
  lambda=0,
  min_child_weight=2,
  objective="multi:softprob",
  eval_metric="mlogloss",
  num_class=2,
  subsample=1,
  colsample_bytree=1
)

# Lets now make use of the cross validation in XGBoost framework to find best nrounds
# We also inspect how the log loss decreases with iterations and use early stopping to stop overfitting
xgb.cv = xgb.cv(
  params = params,
  data = xgb.train,
  nrounds=1000,
  nfold=5,
  showsd=T,
  stratified=T,
  print_every_n=10,
  early_stopping_rounds=20,
  maximize=F
)

# Best Iteration: 90 (convergence)

# Train the XGBoost model with Best iterations
xgb.fit = xgb.train(
  params=params,
  data=xgb.train,
  nrounds=90,
  nthreads=1,
  early_stopping_rounds=10,
  watchlist=list(val1=xgb.train,val2=xgb.test),
  verbose=1
)

# Use XGBoost model to predict test set classifications
xgb.pred = predict(xgb.fit,holdout_x,reshape=T)
xgb.pred = as.data.frame(xgb.pred)
colnames(xgb.pred) = c("M", "B")

# Use the predicted label with the highest probability
xgb.pred$prediction = apply(xgb.pred,1,function(x){colnames(xgb.pred)[which.max(x)]})
xgb.pred$label = colnames(xgb.pred)[holdout_y + 1]

# Show the accuracy of 1st model
result = sum(xgb.pred$prediction==xgb.pred$label)/nrow(xgb.pred)
print(paste("Final Accuracy =",sprintf("%1.2f%%", 100*result)))

# Shows which features were important for training
mat <- xgb.importance (feature_names = colnames(holdout_x),model = xgb.fit)
xgb.plot.importance (importance_matrix = mat[1:10])

#*********************************************************************************
# Lets now try and hypertune some of the many parameters using MLR library

wdbc_train = wdbc_raw[train_indices,]
wdbc_train = wdbc_train[,-1]
wdbc_train$label = factor(wdbc_train$label)
wdbc_test = wdbc_raw[-train_indices,]
wdbc_test = wdbc_test[,-1]
wdbc_test$label = factor(wdbc_test$label)

# Set up tasks for MLE
traintask <- makeClassifTask(data = wdbc_train,target = "label")
testtask <- makeClassifTask(data = wdbc_test,target = "label")

# Set up learning objectives and criteria
lrn <- makeLearner("classif.xgboost", predict.type = "response")
lrn$par.vals <- list(objective="multi:softprob", eval_metric="mlogloss", nrounds=100L, eta=0.1)

# New Parameter list that we want to tune
params2 <- makeParamSet( makeDiscreteParam("booster",values = c("gbtree","gblinear")),
                         makeIntegerParam("max_depth",lower = 3L,upper = 10L), 
                         makeNumericParam("min_child_weight",lower = 1L,upper = 10L), 
                         makeNumericParam("subsample",lower = 0.5,upper = 1), 
                         makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))

rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)

ctrl <- makeTuneControlRandom(maxit = 10L)

# Use parallelised features for speed
parallelStartSocket(cpus = detectCores())

# Put all of the above together to form tuning protocol
mytune <- tuneParams(learner = lrn, 
                     task = traintask, 
                     resampling = rdesc,
                     measures = acc, 
                     par.set = params2, 
                     control = ctrl, 
                     show.info = T)

# Train model
lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)
xgmodel <- mlr::train(learner = lrn_tune,task = traintask)

# Predict
xgpred <- predict(xgmodel,testtask)

# Display model results and performance metrics
confusionMatrix(xgpred$data$response,xgpred$data$truth)



















#check the outliers
#this is the Figure 2.1
p <- ggplot(wdbc_raw,aes(label))+
  geom_bar(width = 0.7,fill=c("#CD3333","#00B2EE"))+
  p_theme+
  labs(x="Diagnosis",y="Count")

#this is the Figure 2.2,2.3,2.4
get_outline <- function(var){
  x_lab <- if(str_detect(var,"_m")){
    str_c("Average of ",str_replace(var,pattern = "_m",replacement = ""),sep = "")
  }else if(str_detect(var,"_w")){
    str_c("Worst of ",str_replace(var,pattern = "_w",replacement = ""),sep = "")
  }else{
    str_c("Standard error of ",str_replace(var,pattern = "_sd",replacement = ""),sep = "")
  }
  p1 <- ggplot(wdbc_raw,aes(x=get(var)))+
    geom_histogram(aes(y=..density..),fill="#00B2EE")+
  stat_function(fun = dnorm,
                  args=list(mean=mean(wdbc_raw[,var,drop=T]),
                            sd=sd(wdbc_raw[,var,drop=T])),
                  col="#CD3333")+
    p_theme+
    labs(y="Density",x=x_lab)
  return(p1)
}
#write a list function then we do not to write repeat codes
pps <- list()
for(i in 1:(ncol(wdbc_raw)-2)){
  pps[[i]] <- get_outline(colnames(wdbc_raw)[2+i])
}
#output of the graphs
(pps[[1]]+pps[[2]])/(pps[[3]]+pps[[4]])
#in order to make the graphs clear, we put these outlier graphs together
(pps[[5]]+pps[[6]]+pps[[7]])/(pps[[8]]+pps[[9]]+pps[[10]])
#there is no requirement for the rank of these numbers
(pps[[11]]+pps[[12]])/(pps[[13]]+pps[[14]])
#just put them one by one
(pps[[15]]+pps[[16]]+pps[[17]])/(pps[[18]]+pps[[19]]+pps[[20]])

(pps[[21]]+pps[[22]])/(pps[[23]]+pps[[24]])
(pps[[25]]+pps[[26]]+pps[[27]])/(pps[[28]]+pps[[29]]+pps[[30]])

describe(wdbc_raw[,3:32])%>%.[,c(2,8,9,3,4)]

#Variable selection
wdbc <- wdbc_raw[,-1]

#Exploratory analysis (including graphs from Figure 3.1 to 3.8)
### average
wdbc$label <- factor(wdbc$label,labels = c("Benign","Malignant"))

#the graph for the mean values of benign and malignant
get_eda <- function(feature){
  p31<- ggplot(wdbc,aes(x=label,y=get(str_c(feature,"m",sep = "_"))))+
    geom_boxplot(width=0.2,fill=c("#008B00","#8B1A1A"),notch = T)+
    geom_violin(alpha=0.2,fill="#7FFF00")+
    p_theme+
    labs(x=" ",y=str_c("Average of ",feature,sep ="" ))
    
#the graph for the worst values of benign and malignant
  p32<- ggplot(wdbc,aes(x=label,y=get(str_c(feature,"w",sep = "_"))))+
    geom_boxplot(width=0.2,fill=c("#008B00","#8B1A1A"),notch = T)+
    geom_violin(alpha=0.2,fill="#7FFF00")+
    p_theme+
    labs(x=" ",y=str_c("Worst of ",feature,sep ="" ))
    
#the scatter plot(x-axis is mean values, y-axis is worst values)  
  p33 <- ggplot(wdbc,aes(x=get(str_c(feature,"m",sep = "_")),
                         y=get(str_c(feature,"w",sep = "_")),
                         size=get(str_c(feature,"sd",sep = "_")),col=label))+
    geom_point(alpha=0.5)+
    p_theme+
    scale_color_manual(values = c("#008B00","#8B1A1A"))+
    labs(x=str_c("The average of ",feature,sep ="" ),
         y=str_c("The worst of ",feature,sep ="" ),
         col="",size="Standard error")
  return((p31+p32)/p33)
}
#get the above graph for every variable we care about
get_eda("radius")
get_eda("perimeter")
get_eda("area")
get_eda("texture")
get_eda("smoothness")
get_eda("concavity")
get_eda("concave")
get_eda("compactness")
get_eda("fd")
get_eda("symmetry")

#the inter-correlation analysis for mean values (Figure 3.9)
wdbc_cor <- wdbc[,2:11]
cor_m <- cor(wdbc_cor)
sig_m <-cor.mtest(cor_m)
corrplot(corr = cor_m,p.mat = sig_m$p,method = "ellipse",
         tl.col = "black",insig = "blank",tl.cex = 0.8)
#the inter-correlation analysis for worst values (Figure 3.10)
wdbc_cor <- wdbc[,22:31]
cor_m <- cor(wdbc_cor)
sig_m <-cor.mtest(cor_m)
corrplot(corr = cor_m,p.mat = sig_m$p,method = "ellipse",
         tl.col = "black",insig = "blank",tl.cex = 0.8)


#Gusta's Part

#_________PCA__________

#normalization

wdbc_raw_N=wdbc_raw

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

for (i in 3:32) {
  wdbc_raw_N[i]=normalize(wdbc_raw_N[i])
}

#PCA to all the features
pca = prcomp(wdbc_raw_N[3:32],center = TRUE,scale=TRUE)
pca
summary(pca)

#PCA to mean features
mean_pca <- prcomp(wdbc_raw_N[,c(3:12)], scale = TRUE)
mean_pca
summary(mean_pca)

#PCA to standard deviation features
sd_pca <- prcomp(wdbc_raw_N[,c(13:22)], scale = TRUE)
sd_pca
summary(sd_pca)

#PCA to worst case features
worst_pca <- prcomp(wdbc_raw_N[,c(23:32)], scale = TRUE)
worst_pca
summary(worst_pca)


wdbc_pca1 = as.data.frame(predict(pca, wdbc_raw_N))
wdbc_pca1$label = wdbc_raw$label

wdbc_pca2 = as.data.frame(predict(mean_pca, wdbc_raw_N))
wdbc_pca2$label = wdbc_raw$label

wdbc_pca3 = as.data.frame(predict(sd_pca, wdbc_raw_N))
wdbc_pca3$label = wdbc_raw$label

wdbc_pca4 = as.data.frame(predict(worst_pca, wdbc_raw_N))
wdbc_pca4$label = wdbc_raw$label

#__________TRAIN-TEST DATA_______________

#1)
#seperate the 30% of the data
set.seed(123)
test_indices = sample(1:nrow(wdbc_pca1),
                      size = round(0.3*nrow(wdbc_pca1 )),
                      replace = FALSE)

#the train data
train = wdbc_pca1 [-test_indices,]
#the test data
test = wdbc_pca1 [test_indices,]

#2)

#seperate the 30% of the data
test_indices = sample(1:nrow(wdbc_pca2),
                      size = round(0.3*nrow(wdbc_pca2 )),
                      replace = FALSE)

#the train data
train = wdbc_pca2 [-test_indices,]
#the test data
test = wdbc_pca2 [test_indices,]

#3)

#seperate the 30% of the data
test_indices = sample(1:nrow(wdbc_pca3),
                      size = round(0.3*nrow(wdbc_pca3 )),
                      replace = FALSE)

#the train data
train = wdbc_pca3 [-test_indices,]
#the test data
test = wdbc_pca3 [test_indices,]

#4)

#seperate the 30% of the data
test_indices = sample(1:nrow(wdbc_pca4),
                      size = round(0.3*nrow(wdbc_pca4 )),
                      replace = FALSE)

#the train data
train = wdbc_pca4 [-test_indices,]
#the test data
test = wdbc_pca4 [test_indices,]

#MODEL







#WE CAN ADD THE MODELS TO HERE








#MODEL EVALUATION

library(caret)
caret::confusionMatrix(y_pred, y_act, positive="1", mode="everything")

#if you find a probability and use the cutoff to change the perception to say someone malignant
confusion = confusionMatrix(actual=type_test,predicted=predicted_probs[,2],cutoff=0.5)
confusion

#if you directly predict true or false you can basicaly create the table with that function
#confusion = table(predicted, type_test); confusion_matrix
#confusion

rownames(confusion) = c("predicted_benign","predicted_malignant")
colnames(confusion) = c("benign","malignant")
confusion

#this the number of person who have benign tumor and predicted correctly
true_benign = confusion["predicted_benign","benign"]
true_benign

#this is the number of person who have malignant tumor and predicted correctly 
true_malignant = confusion["predicted_malignant","malignant"]
true_malignant

#this is the number of person who have benign but predicted as malignant
false_malignant = confusion["predicted_malignant","benign"]
false_malignant

#this is the number of person who have benign but predicted as malignant
false_benign = confusion["predicted_benign","malignant"]
false_benign

#error rate
misclassification_rate = (false_malignant + false_benign) / sum(confusion)
misclassification_rate
#accuracy rate
success_rate=(true_benign+true_malignant)/sum(confusion)
success_rate
#precision is what percent of the malignant diagnosis are correct
precision=true_malignant/(true_malignant+false_malignant)
precision
#the sensitivity of the model, the percentage of the correct malignant diagnosis in the total malignant cases
#correct malignant diagnosis/total malignant cases
#what percentage of the malignants tumors that we could predict
recall = true_malignant / (true_malignant + false_benign)
recall
#the selectivity of the model, the rate of correctly diagnosed benign in the total benign cases
#correct benign diagnosis/total benign cases
#what percentage of the benign tumors that we could predict
selectivity = true_benign / (true_benign + false_malignant)
selectivity


#KAPPA STATISTIC 

#The total number of test instances is:..
#The total number of correct diagnose is:...
#The accord of the model with the test data is the accuracy rate whic we have
#calculated before as success_rate but it can be affacted by random predictions 
#and we want to see the consistency of the method with the data without the effect
#of the random prediction by chance

#The total number of prediction of bening is : 
pred_benign=(true_benign+false_benign)/(true_benign+false_benign+true_malignant+false_malignant)
#The total number of prediction of malignant is :
pred_malignant=(true_malignant+false_malignant)/(true_benign+false_benign+true_malignant+false_malignant)
#The total number of malignant cases is:
malignant_case=(true_malignant+false_benign)/(true_benign+false_benign+true_malignant+false_malignant)
#The total number of bening cases is:
bening_case=(true_benign+false_malignant)/(true_benign+false_benign+true_malignant+false_malignant)


#the probability of randomly selecting the benign
random_benign=pred_benign*bening_case

#the probability of randomly selecting the malignant
random_malignant=pred_malignant*malignant_case

#the total random accordance probability of the model and test dataset is;
total_random=random_benign+random_malignant

#tha value of kappa;
kappa=(success_rate-total_random)/(1-total_random)
kappa

#the shortest path to calculate the kappa but we have created our
#own way to understand the logic
kappa(actual=type_test,predicted=predicted_probs[,2],cutoff=0.5)

#F1-SCORE
#combination of recall and precision
F1_Score = (2 * precision * recall) / (precision + recall)
F1_Score

#ROC CURVE
library(InformationValue)
InformationValue::plotROC(type_test, predicted_probs[,2])
roc=InformationValue::AUROC(type_test, predicted_probs[,2])
roc
#the optimal cut-off(threshold to call a tumour malignant) to maximize the recall and selectivity.
optimalCutoff(actuals = type_test, predictedScores =predicted_probs[,2] )

#COMPARISON OF THE METHODS AND MEASUREMENT RATES
comparison = data.frame("misclassification" = c(misclassification_rate),
                        "precision" = c(precision,precision,precision),
                        "recall" = c(recall,recall,recall),
                        "selectivity"=c(selectivity,selectivity,selectivity),
                        "kappa statistic"=c(kappa,kappa,kappa),
                        "F1 Score"=c(F1_Score,F1_Score,F1_Score),
                        "ROC"=c(roc,roc,roc))

#we can add new methods
rownames(comparison) = c("Method1","Method2","Method3")
comparison


