PracMachLearn
========================================================
author: Nicholas Paul Hartman
date: 18 August 2014

Overview
========================================================

This presentation is my submission for the Write-up portion of the course project
for the Coursera Practical Machine Learning course offered by Johns Hopkins department
of Biostatistics.  Our objectives are:

- Evaluate a provided dataset, exploring what data to include and not include
- Build a prediction model from the training portion of those data
- Speak to an estimate on out of sample error
- Apply our model to the provided test data and submit those predictions separately

First Steps
========================================================

First of all, we'll need to load in the data and arrange it for further analysis.


```r
## Load needed libraries for likely models
require(caret)
require(randomForest)

## Read the data into R.  This requires source files in working directory.
train <- read.csv("pml-training.csv",header=TRUE)
test <- read.csv("pml-testing.csv",header=TRUE)
```

Initial Evaluation
========================================================

So now let's pop the hood.


```r
##  What are these puppies shaped like?
dim(train);dim(test)
```

```
[1] 19622   160
```

```
[1]  20 160
```

160 variables is... ambitious.  Let's check if life can be made easier by scratching variables that are completely NA in the test set, and thus unusable for prediction, or any variables w/ a single value

First Pass on Filtering
========================================================


```r
## Gathering any empty columns
duds <- sapply(test, function(test)all(is.na(test)))
duds <- as.numeric(paste(which(duds)))

## Now gathering any columns with only one value
levs <- sapply(1:160,function(x) levels(test[,x]))
only <- lapply(levs,function(x) length(x)==1)
which(only==TRUE)
```

```
[1] 6
```

Implement First Pass
========================================================

In the test set, it appears that the new_window variable has only a single value.  We can subset the training set to cut down the necessary calculations then.


```r
clean.train <- subset(train,train$new_window=='no')
```

We can also drop columns from the new "clean" train set which are all NAs in the test set


```r
clean.train <- clean.train[,-duds]
```

Other Low-Hanging Fruit
========================================================

Let's now remove variables that common sense dictate have no predictive power; specifically Name, date, time and window variables which have nothing to do with the exercise movements themselves


```r
clean.train <- clean.train[,-c(1:7)]
```

Now let's check for near zero variability

```r
near.zero <- nearZeroVar(clean.train,saveMetrics=FALSE)
length(near.zero)
```

```
[1] 0
```

Plan Now
========================================================

OK cool, not much obvious fat to cut.  Browsing the names of the remaining variables (not performed here - just call names(clean.train)) we're down to the same 13 measurements taken from 4 different locations on the subject plus the classe variable we're aiming to predict.

We're predicting category, and my first thought there is foresting.  Given the fairly manageable size of the data, at this point I'm opting for Random Forest.

Why?  Preprocessing is less of a concern, and while the calc time is long, the accuracy should be very high (important for the other submission portion of this assignment) and I'm not as concerned about interpretability of the output.

Subsetting True.Train and CV.Train
========================================================

So all that said, I'm going to consider the training set sufficiently boiled down for our purposes.  Let's split this up for so we've got a pre-test test set.


```r
sub <- createDataPartition(y=clean.train$classe,p=.6,list=FALSE)
true.train <- clean.train[sub,]
test.train <- clean.train[-sub,]
```

Before running the RF train - a heavy duty calculation - Let's clean up the environment. Additional resources = nice.


```r
remove(duds,levs,near.zero,only,
       clean.train,sub,train)
```

Final Thoughts on Model
========================================================

Cool.  Now let's first try building a random forest model off the true.train data, see where that gets us with the cv.train set.  As indicated earlier I've decided to use RF because of the categorical nature of the predicted values, RF's strong ability to predict with high accuracy, its ability to handle outliers and other weirdness, and the cross validation inherent in the nature of the procedure.  Please see more here for explanation:

**stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#ooberr**

A Note on our Random Forest Cross-Val
========================================================

We're going to adjust trainControl to switch train's defualt cross-val method from bootstrapping 25 samples and taking FOREVER to just 5 samples.  This is selected purely for expediency - I know I'm sacrificing on model, but the first time I ran this at 25, it took ~12 hours to complete on an MB Pro @ 2.9GHz i7 w/ 8GB 1600Mhz DDR3 with nothing running in the background and a clean environment!  I'm also confident that we'll get a solid reading with the variables we have (a bit meta, but I don't think the instructors would assign a dataset from which a decent prediction couldn't be drawn).

Fit That Model!
========================================================

OK so here we go!


```r
modFit <- train(classe~ .,data=true.train,method="rf",prox=TRUE,
                trControl = trainControl(number = 5))
```

Accuracy
========================================================

This method still takes an hour or two, but a call of

```r
confusionMatrix(test.train$classe,
                predict(modFit,test.train))$overall
```

```
      Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
        0.9923         0.9903         0.9901         0.9941         0.2862 
AccuracyPValue  McnemarPValue 
        0.0000            NaN 
```
Shows we're at 99.23% accuracy.  That's pretty good.  With regard to expected error:

```r
modFit$finalModel
```

Error
========================================================


```

Call:
 randomForest(x = x, y = y, mtry = param$mtry, proximity = TRUE) 
               Type of random forest: classification
                     Number of trees: 500
No. of variables tried at each split: 27

        OOB estimate of  error rate: 0.96%
Confusion matrix:
     A    B    C    D    E class.error
A 3276    4    2    0    1    0.002132
B   20 2200   10    1    0    0.013895
C    0   12 1989   11    0    0.011431
D    0    4   28 1854    3    0.018528
E    0    1    6    8 2102    0.007085
```
***
shows OOB estimate of error at 0.96%.  I think that's a pretty acceptable expected out of sample error rate.

Final Run
========================================================

So, it's show time.  You'll have to take my word for it, but running:


```r
kick.ass <- predict(modFit,newdata=test)
```

and then uploading the answers against the submission assignment came in with 100% correct results.  Call it a day!
