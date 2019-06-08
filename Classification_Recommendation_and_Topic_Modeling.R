load("sngame.Rda")
str(sngame)
head(sngame)

library(psych)
library(car)

#  MULTIVARIATE REGRESSION MODEL
scatterplotMatrix(sngame)
pairs(~ sngame$game.min+sngame$sn.conn+sngame$sn.min
      +sngame$game.purchase+sngame$age+sngame$gender+sngame$edu+sngame$salary)

sngame2 <- sngame
sngame2$gender <- Recode(sngame2$gender, "'Female'=1; 'Male'=2", as.factor = F)
unique(sngame2$edu) #Elementary School High School University
sngame2$edu <- Recode(sngame2$edu, "'Elementary School'=1; 'High School'=2; 'University'=3", as.factor = F)
head(sngame2)
str(sngame2)
cor(sngame2) #displaying the correlations

pairs.panels(sngame, 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses
)


#The scatterplot diagrams, correlation coefficients, and p-values suggest that the variables significantly
#affecting the amount of purchases made on game are:-
#game.min,
#sn.min (minutes spent on the social network),
#salary (user monthly salary- though slightly less significantly)


game_fit2 <- lm(sngame$game.purchase~sngame$sn.min+sngame$game.min+sngame$salary)
summary(game_fit2)
confint(game_fit2)

#evaluating the variance inflation factors
vif(game_fit2)

plot(game_fit2)
#Evaluating visually for linearity
crPlots(game_fit2)
#Evaluating visually for Normality
qqPlot(game_fit2, labels=row.names(sngame), id.method="identify",
       simulate=TRUE, main="Q-Q Plot")

#         Prediction Attempt:-
ourdata<- subset(sngame[,c(1,2,3,4,5,6,7,8)])
ourdata$game_purchase_guess <- predict(game_fit2, newdata=ourdata, type="response")
print(head(ourdata))


haberman_data<- read.csv("haberman.csv", sep = ",", header = FALSE)
str(haberman_data)
head(haberman_data)
#checking the median, mean and basic descriptive statistics
summary(haberman_data)

#making the survival outcome a factor
haberman_data$V4 <- as.factor(haberman_data$V4)
str(haberman_data)
#checking the frequency table according to patient survival
table(haberman_data$V4)

library(rpart)
library(rpart.plot)
library(caTools)


#Attribute Information:
#1. Age of patient at time of operation (numerical)
#2. Patient's year of operation (year - 1900, numerical)
#3. Number of positive axillary nodes detected (numerical)
#4. Survival status (class attribute)
#      1 = the patient survived 5 years or longer
#      2 = the patient died within 5 year

# Evaluation function for evaluating the "Accuracy", "Sensitivity", "Specificity", "Precision", "NPP" using the confusion matrix
eval_function <- function(tn, fn, fp, tp){
  accuracy <- (tp + tn) / (tp + tn + fn + fp)
  sensitivity <- tp / (tp + fn)
  specificity <- tn / (fp + tn)
  precision <- tp / (tp + fp)
  npp <- tn / (tn + fn)
  res <- c(accuracy, sensitivity, specificity, precision, npp)
  names(res) <- c("Accuracy", "Sensitivity", "Specificity", "Precision", "NPP")
  res
}


#  CLASSIFICATION TREE MODEL
#fixing the randomization in the split
set.seed(142941)

split <- sample.split(haberman_data$V4, SplitRatio = 0.70) #spliting the data in test and training sets (30:70)
split
haberman_training_set <- subset(haberman_data, split == TRUE)
haberman_test_set <- subset(haberman_data, split == FALSE)

summary(haberman_training_set)
summary(haberman_test_set)
#Building the classification tree using Age, Year of operation, Number of positive axillary nodes detected).
#tree_cancer <- rpart(V4 ~ V1 + V2 + V3, data = haberman_training_set, method="class")
#the results were not quite good, hence a penalty included for FN

penalty <- matrix(c(0,15,100,0), nrow = 2, byrow = TRUE)

tree_cancer <- rpart(V4 ~ V1 + V2 + V3, data = haberman_training_set,
                     method="class", cp=0.005, minbucket = 3, parms = list(loss=penalty))

prp(tree_cancer)
rpart.plot(x = tree_cancer, yesno = 2, type = 0, extra = 0)

summary(tree_cancer)

Predict_cancer <- predict(tree_cancer, newdata = haberman_training_set, type = "class")

#creating the confusion matrix fro training data
tab <- table(haberman_training_set$V4, Predict_cancer)
tab
#Trying the model on the test data
Predict_cancer_test <- predict(tree_cancer, newdata = haberman_test_set, type = "class")

#creating the confusion matrix for test data
tab_test <- table(haberman_test_set$V4, Predict_cancer_test)
tab_test
#calculating the classification performance for test data
my_evaluation_test <- eval_function(tab_test[1,1], tab_test[1,2], tab_test[2,1], tab_test[2,2])
my_evaluation_test
#Accuracy Sensitivity Specificity   Precision         NPP 


##Classification with Neural Network
library('nnet')
library('neuralnet')
library(caTools)
library('NeuralNetTools')
library('devtools')

#                   USING NEURALNET
bank_data<- read.csv("bank-full.csv", sep = ";", header = TRUE)
bank_data$y <- Recode(bank_data$y, "'yes'=1; 'no'=0", as.factor = F)
indx <- sapply(bank_data[,-17], is.factor)
bank_data[indx] <- lapply(bank_data[indx], function(x) as.numeric(as.factor(x)))
bank_data[,1:(ncol(bank_data) - 1)] <- scale(bank_data[,1:(ncol(bank_data) - 1)])
bank_data$y <- as.integer(bank_data$y)
str(bank_data)
head(bank_data)
summary(bank_data)
set.seed(1985)
new_split <- sample.split(bank_data$y, SplitRatio = 0.75)
training_set <- subset(bank_data, new_split == TRUE)
test_set <- subset(bank_data, new_split == FALSE)
dataframef <- reformulate(setdiff(colnames(training_set), "y"), response="y")
neural_bank <- neuralnet(dataframef, training_set, hidden=c(8,4))
plot(neural_bank)
predict_test <- compute(neural_bank, test_set[,1:(ncol(test_set)-1)])
tab <- table(test_set$y, predict_test$net.result > 0.2)
tab
evalresult <- eval_class(tab[1,1], tab[1,2], tab[2,1], tab[2,2])
print(evalresult)
#Accuracy  Sensitivity  Specificity    Precision          NPP 
