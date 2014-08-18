## Hartman, Nicholas Paul
## nicholas.paul.hartman@gmail.com
## Practical Machine Learning Course Project
## Script(s) for initial data load, evaluation, preprocessing

## Let's load some needed libraries for likely models
require(caret)
require(randomForest)

## Read the data into R
## Please note, this requires source files in working directory
train <- read.csv("pml-training.csv",header=TRUE)
test <- read.csv("pml-testing.csv",header=TRUE)

## Before we do any model building, 160 variables is... ambitious.
## Let's check if life can be made easier by scratching variables that are completely NA
## in the test set, and thus unusable for prediction, or any variables w/ a single value

duds <- sapply(test, function(test)all(is.na(test)))
duds <- as.numeric(paste(which(duds)))

levs <- sapply(1:160,function(x) levels(test[,x]))
only <- lapply(levs,function(x) length(x)==1)
which(only==TRUE)

## In the test set, it appears that the new_window variable has only a single value.
## We can subset the training set to cut down the necessary calculations then.
clean.train <- subset(train,train$new_window=='no')

## Drop columns from new "clean" train set which are all NAs in the test set
clean.train <- clean.train[,-duds]

## Remove variables that common sense dictate have no predictive power,
## specifically Name, date, time and window variables which have nothing to do
## with the exercise movements themselves
clean.train <- clean.train[,-c(1:7)]

## Now let's check for near zero variability
near.zero <- nearZeroVar(clean.train,saveMetrics=FALSE)
length(near.zero)

## OK cool, not much fat to cut.  We're predicting category, so I'm not
## interested as much in linear modeling, I won't worry about centering/scaling.
## I'm leaning Random Forest, so I'm also not concenred about other preprocessing.
## I'm going to consider the training set sufficiently boiled down for our purposes,
## let's split this up for so we've got a pre-test test set.
sub <- createDataPartition(y=clean.train$classe,p=.6,list=FALSE)
true.train <- clean.train[sub,]
test.train <- clean.train[-sub,]

## One last thing before running the train() command - this is a heavy duty calculation
## and it's worth cleaning up the environment for resource availability.
## Also, generally having a clean workbench.
remove(duds,levs,near.zero,only,clean.train,sub,train)

## Cool.  Now let's first try building a random forest model off the true.train data,
## see where that gets us with the cv.train set.  I've decided to use RF because
## of its strong ability to predict with good accuracy, its ability to handle
## outliers and other weirdness, and the cross validation inherent in the nature
## of the procedure.  Please see more here for explanation:

## http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#ooberr

## Furthermore, I'm less interested
## in result interpretability right now than a solid, accurate answer.
## We're going to adjust trainControl to switch train's defualt cross-val method
## from bootstrapping 25 samples and taking FOREVER to just 5 samples.  This
## is selected purely for expediency - I know I'm sacrificing on model, but the 
## first time I ran this at 25, it took ~12 hours to complete on an
## MB Pro @ 2.9GHz i7 w/ 8GB 1600Mhz DDR3 and nothing running in the background!
## I'm also confident that we'll get a solid reading with 52 predictor variables,
## 13 different readings from 4 different sensor locations for the exercises.
modFit <- train(classe~ .,data=true.train,method="rf",prox=TRUE,
                trControl = trainControl(number = 5))

## This method still takes an hour or two, but a call of
confusionMatrix(test.train$classe,predict(modFit,test.train))

## Shows we're at 99.21% accuracy.  That's pretty good.  With regard to expected error:
modFit$finalModel

## shows OOB estimate of error at 0.94%.  "Out of Bag" by definition is the expected
## error outside of the sample.  I think that's a pretty acceptable error rate.

## So, it's show time.  You'll have to take my word for it, but running:
kick.ass <- predict(modFit,newdata=test)

## and then uploading the answers against the submission assignment came in with 100%
## correct results.  Call it a day!