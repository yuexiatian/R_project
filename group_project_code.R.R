## Emily Wijker - u214079
## Patricija Milaseviciute - u746955
## Yuexia Tian - u976100
## Anouk Heemskerk - u684364


## Load packages ---------------------------------------------------------------
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)
# our own package
install.packages("randomForest")
library(randomForest)

## Get citations ---------------------------------------------------------------
citation("dplyr")
citation("tidyr")
citation("ggplot2")
citation("caret")
citation("randomForest")

## Load data -------------------------------------------------------------------
diabetes_data <- read.csv("input/diabetes_data_upload.csv", stringsAsFactors = T,
                          na = "NA")


## Step 1: Exploratory Data Analysis -------------------------------------------

summary(diabetes_data)
str(diabetes_data)

# change column names to lower case

names(diabetes_data) <- tolower(names(diabetes_data))

# check for missing data

which(is.na(diabetes_data))  # 0 missing

# check age distribution

table(diabetes_data$age)

ggplot(diabetes_data, aes(x = age)) +
  geom_bar()

# boxplot of age showing outliers

boxplot(diabetes_data$age)

# outlier deletion

diabetes_data <- diabetes_data[-which(diabetes_data$age > 80),]

# age divided by gender

ggplot(data = diabetes_data, aes(x = age)) +
  geom_bar() +
  facet_grid(. ~ gender)

# bar plot of gender distribution per class

ggplot(data = diabetes_data, aes(x = class, fill = gender)) +
  geom_bar()

# summary showing positive class statistics

symptom_summary <- diabetes_data %>%
  filter(class == "Positive")

summary(symptom_summary)


## Step 2: Fit model using knn -------------------------------------------------

# split into a train and test set
set.seed(1)
trn_index <- createDataPartition(y = diabetes_data$class, p = 0.70, 
                                 list = FALSE)
trn_diabetes <- diabetes_data[trn_index, ]
tst_diabetes <- diabetes_data[-trn_index, ]

# fit knn model
set.seed(1)
diabetes_knn <- train(class ~ . - age - gender ,
                         method = "knn", data = trn_diabetes,
                         trControl = trainControl(method = "cv", number = 5,
                                                  returnResamp = "all"),
                         tuneGrid = data.frame(k = seq(3, 10, by = 1))
)

diabetes_knn

# writing a plot function for knn models
plot_knn_results <- function(fit_knn) {
  ggplot(fit_knn$results, aes(x = k, y = Accuracy)) +
    geom_bar(stat = "identity") +
    scale_x_discrete("value of k", limits = fit_knn$results$k) +
    scale_y_continuous("accuracy")
}

plot_knn_results(diabetes_knn)
# best model is k = 3, accuracy = 0.906

## evaluating Performance on the Test Set
set.seed(1)
predicted_knn <- predict(diabetes_knn, tst_diabetes)

knn_confM <- caret::confusionMatrix(predicted_knn, 
                                    as.factor(tst_diabetes$class))
knn_confM

summary(diabetes_knn)

## Step 3: Fit model using logistic regression ---------------------------------

# fit logistic regression model (AUC)
set.seed(1)
diabetes_lgr_AUC <- train(class ~ . - age - gender, method = "glm",
                          family = binomial(link = "logit"),
                          data = trn_diabetes,
                          trControl = trainControl(method = "cv", number = 5,
                                                   classProbs = TRUE,
                                                   summaryFunction = prSummary
                         ), metric = "AUC"
)

diabetes_lgr_AUC
summary(diabetes_lgr_AUC)

# fit logistic regression model (RECALL)
set.seed(1)
diabetes_lgr_REC <- train(class ~ . - age - gender, method = "glm", 
                          family = binomial(link = "logit"), 
                          data = trn_diabetes,
                          trControl = trainControl(method = "cv", number = 5,
                                                  classProbs = TRUE,
                                                  summaryFunction = prSummary
                         ), metric = "Recall"
)

diabetes_lgr_REC
summary(diabetes_lgr_REC)

# fit logistic regression model (ACCURACY)
set.seed(1)
diabetes_lgr_ACC <- train(class ~ . - age - gender, method = "glm", 
                          family = binomial(link = "logit"), 
                          data = trn_diabetes,
                          trControl = trainControl(method = "cv", number = 5,
                                                   classProbs = TRUE)
                          )

diabetes_lgr_ACC
summary(diabetes_lgr_ACC)

# create confusion matrix
# we want to maximize true positives so will use the recall model
set.seed(1)
predicted_lgr <- predict(diabetes_lgr_REC, tst_diabetes)

lgr_confM <- caret::confusionMatrix(predicted_lgr,
                                    as.factor(tst_diabetes$class))

lgr_confM

## Step 4: Our third ML / stats technique --------------------------------------

# fit Random Forest model
diabetes_rf1 <- randomForest(class ~ . - age - gender,data = trn_diabetes, 
                            importance = TRUE)

diabetes_rf1

# Fine tuning parameters of Random Forest model
diabetes_rf2 <- randomForest(class ~ . - age - gender, data = trn_diabetes, 
                             ntree = 500, mtry = 6, importance = TRUE)
diabetes_rf2

# Predicting on train set
set.seed(1)
predTrain <- predict(diabetes_rf2, trn_diabetes, type = "class")

# Predicting on test set
predTest <- predict(diabetes_rf2, tst_diabetes, type = "class")

# Checking classification accuracy
table(predTest, tst_diabetes$class)  
mean(predTest == tst_diabetes$class)

# creating confusion matrix
rf_confM <- caret::confusionMatrix(predTest,
                                   as.factor(tst_diabetes$class))
rf_confM

# Checking important variables
importance(diabetes_rf2)        
varImpPlot(diabetes_rf2)   

