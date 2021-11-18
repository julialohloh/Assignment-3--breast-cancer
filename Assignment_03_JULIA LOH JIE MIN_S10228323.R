# =====================================================================================================================
# Name: JULIA LOH JIE MIN
# MACHINE LEARNING FOR BIOMEDICAL AND HEALTHCARE APPLICATIONS 
# Ngee Ann Polytechnic 
# Assignment 3
# =====================================================================================================================

# ---------------------------------------------------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------------------------------------------------
# Remove existing objects, start with clean workspace
rm(list=ls())

# --------------------------------------------------------------------------------------------------------------------
# LOAD DATA AND INSPECT DATA STRUCTURE
# --------------------------------------------------------------------------------------------------------------------
# QUESTION 1
library(pacman)
p_load(tidyverse, mice, caret, pROC,kernlab, randomForest, factoextra, ggfortify, fpc)

# Import data
d <- read.csv("Breast Cancer.csv")

# QUESTION 1a : number of rows and columns
# Inspect data structure, data types and values
glimpse(d)
# QUESTION 1b: ratio of malignant to benign cases
round(prop.table(table(d$diagnosis)), 2)

# QUESTION 2a: remove the column "id" from the data set
d = subset(d, select = -c(id))
glimpse(d)

# QUESTION 2b: convert the data type of the column "diagnosis" to factor
d$diagnosis <- as.factor(d$diagnosis)

# QUESTION 2C: check for missing data and remove records that contain missing data
d <- na.omit(d)
md.pattern(d, rotate.names = TRUE)

# --------------------------------------------------------------------------------------------------------------------
# PARTITION DATASET INTO TRAINING SET AND TEST SET
# --------------------------------------------------------------------------------------------------------------------

# QUESTION 3: partitioning into training (70%), test (30%)
set.seed(123)
index.trgSamples <- createDataPartition(d$diagnosis, p=0.7, times = 1, list=F)
trgSamples <- d[index.trgSamples,]
tstSamples <- d[-index.trgSamples,]

# Check the number of rows
nrow(trgSamples)
nrow(tstSamples)


# --------------------------------------------------------------------------------------------------------------------
# BUILD MACHINE LEARNING MODELS
# --------------------------------------------------------------------------------------------------------------------

# QUESTION 4: using seven-fold cross validation
fit.control <- trainControl(
   method ="cv", 
   number =7,   
   classProbs = TRUE
)

# QUESTION 4a: Logistic Regression
model1 <- train(diagnosis ~ ., data = trgSamples,
               method = "glm",
               trControl = fit.control,
               family = "binomial"
)

# QUESTION 4b: Support Vector Machine (Radial Kernel)
model2 <- train(diagnosis ~ ., data = trgSamples,
                         method = "svmRadial",
                         trControl = fit.control,
                         preProcess=c("center","scale")
)

# QUESTION 4c: Random Forest
model3 <- train(diagnosis ~ ., data = trgSamples,
                  method = "rf",
                  trControl = fit.control,
                  preProcess=c("center","scale")
)

# QUESTION 4d: Neural Net
model4 <- train(diagnosis ~ ., data = trgSamples,
                method = "nnet",
                trControl = fit.control,
                preProcess=c("center","scale")
)


# --------------------------------------------------------------------------------------------------------------------
# PREDICT
# --------------------------------------------------------------------------------------------------------------------

# QUESTION 5
doPrediction <- function(m, nameOfModel, tst) {
   
   # Arguments
   # m - model
   # nameOfModel - character string describing the model
   # tst - test samples to run the prediction on
   # ---------------------------------------------------

   # predict
   predicted.prob <- predict(m, newdata = tst, type = "prob")
   # convert  probabilities to outcomes using threshold of 0.5
   predicted.outcomes <- factor(ifelse(predicted.prob$M >= 0.5,"M","B"))
   # generate confusion matrix
   cm <- confusionMatrix(predicted.outcomes,tst$diagnosis,positive='M')
   # generate normal roc curve
   r.actual <- roc(tst$diagnosis,predicted.prob$M) # generate ROC curve
   # generate smooth roc curve
   r.smooth <- roc(tst$diagnosis,predicted.prob$M, smooth = TRUE)
   plot(r.smooth, lwd = 1, lty = "dashed", col = "darkgray", 
        main = sprintf("Model = %s, AUC = %0.3f",nameOfModel,r.smooth$auc), cex.main = 0.8
   )
   par(new = TRUE)
   plot(r.actual, lwd = 1, col = "blue")
   
   # find optimum threshold and corresponding model performance indicators
   c <- coords(r.actual,
               x = "best", 
               ret=c("threshold", "accuracy", "recall", "precision", "specificity", "sensitivity"), 
               best.method = "closest.topleft")
   points(c$specificity, c$sensitivity, pch=19, col="red")
   text(
      x = c$specificity, 
      y = c$sensitivity, 
      pch=19, 
      col="red", 
      labels = sprintf("Thr = %3.3f",c$threshold), 
      pos = 2,
      cex = 0.8)
   # compare performance indicators, put into data.frame for ease of display
   metricTable <- data.frame(
      model = nameOfModel,
      def.Accuracy = cm$overall['Accuracy'],
      def.Sensitivity = cm$byClass['Sensitivity'],
      def.Specificity = cm$byClass['Specificity'],
      def.Precision = cm$byClass['Precision'],
      def.F1 = 2*cm$byClass['Sensitivity']*cm$byClass['Precision']/(cm$byClass['Sensitivity']+cm$byClass['Precision'])
   )
   row.names(metricTable) <- NULL
   return(metricTable)
}

par(mfrow = c(2,2)) # plot for 2 x 2 charts

# QUESTION 6: Prediction results
mp1 <- doPrediction(model1, "Logistic Regression", tstSamples)
mp2 <- doPrediction(model2, "SVM (Radial)", tstSamples)
mp3 <- doPrediction(model3, "Random Forest", tstSamples)
mp4 <- doPrediction(model4, "nnet", tstSamples)

# Combine for ease of comparison of different models
combined.metricTable <- rbind(mp1, mp2, mp3, mp4)
combined.metricTable



# --------------------------------------------------------------------------------------------------------------------
# Perform PCA
# --------------------------------------------------------------------------------------------------------------------
# PCA works only with numeric variables. 
d.numeric <- d %>% select_if(is.numeric)
glimpse(d.numeric)
d.numeric <- na.omit(d.numeric)

#PCA scaling and standardization
p <- prcomp(d.numeric, center = TRUE, scale = TRUE) 

# --------------------------------------------------------------------------------------------------------------------
# Inspect and visualize PCA
# --------------------------------------------------------------------------------------------------------------------
p$rotation

summary(p)

# QUESTION 11: Create PCA data
d.pca <- data.frame(p$x) %>% select(PC1:PC10) %>% data.frame(diagnosis=d$diagnosis,.)
glimpse(d.pca)

# Original data 
d[1,]

# Transformed data
d.pca[1,]

# QUESTION 12: Partitioning data based on training (70%), test (30%)
set.seed(123)
#partitioning is 70% (p=0.7)
index.trgSamples <- createDataPartition(d.pca$diagnosis, p=0.7, times = 1, list=F)
pca.trgSamples <- d.pca[index.trgSamples,]
pca.tstSamples <- d.pca[-index.trgSamples,]

# QUESTION 14: using seven-fold cross validation
fit.control <- trainControl(
   method ="cv", # use cross-validation
   number =7,   # 7-fold cross validation
   classProbs = TRUE # compute class probabilities
)

# QUESTION 15: Prediction for Model 5
model5 <- train(diagnosis~ ., data = pca.trgSamples,
                method = "glm",
                trControl = fit.control,
                family = "binomial" # logistic regression
)

mp5 <- doPrediction(model5, "Logistic Regression (pca)", pca.tstSamples)
combined.metricTable <- rbind(mp1, mp5)
combined.metricTable


# ====================================================================================================================
# END OF ASSIGNMENT 3
# ====================================================================================================================


