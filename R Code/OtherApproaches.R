# Only source data preparation script if data are missing
if(!("all.dat" %in% ls())) source("./DataPrep.R")

# Log loss measures the amount of total "surprise" going from the predictions
# to the actual data, 30% prob of 1 with 1 true > 94% prob of 1 with 1 true

# Accuracy measures the total amount of times true positives and true
# negatives were detected amongst all of the cases, proportion between 0 and 1

# Let's see how well we can model these data with a RF
lib_load("randomForest")
lib_load("MLmetrics")
lib_load("caret")

rfModel <- randomForest(as.factor(Occupancy) ~ .,
                        data = all.dat$training,
                        ntree = 1e4,
                        mtry = 2)

# RF Log loss
rfPred <- predict(rfModel, 
                  all.dat$validation, type = "Prob")[,2]

# 0.1380896 #
LogLoss(y_pred = rfPred, 
        y_true = all.dat$validation$Occupancy)

# RF Accuracy
rfPred2 <- predict(rfModel, 
                   all.dat$validation, type = "class")

# 0.9505 #
round(confusionMatrix(data = rfPred2, 
                reference = all.dat$validation$Occupancy)$overall,4)

# Null model log loss
# 0.716812 #
LogLoss(y_pred = mean(all.dat$training$Occupancy), 
        y_true = all.dat$validation$Occupancy)

# Null model accuracy, find most occuring Occupancy (0 or 1)
# 0.6353 #
null_guess <- rep(as.numeric(names(table(all.dat$validation$Occupancy))[1]),
                  length(all.dat$validation$Occupancy))

round(confusionMatrix(data = null_guess, 
                reference = all.dat$validation$Occupancy)$overall,4)

# or equivalently
1-mean(all.dat$validation$Occupancy)

# Logistic regression, unregularized
# Similar functional form to the sigmoid function, they're cousins
logreg <- glm(as.factor(Occupancy) ~.,
              data = all.dat$training,
              family = binomial(link = "logit"))

# Log reg log loss
logpred <- predict(logreg, all.dat$validation, type = "response")

# 0.08585132 #
LogLoss(y_pred = logpred, y_true = all.dat$validation$Occupancy)

# Log reg accuracy
logpred2 <- round(logpred, digits = 0)

# 0.9741 #
round(confusionMatrix(data = logpred2, 
                reference = all.dat$validation$Occupancy)$overall,4)

print(coef(logreg), digit = 22)

# (Intercept)          Temperature             Humidity                Light                CO2                   HumidityRatio 
# -4.33202811558247181 -0.91922576450224724  1.71425273557116387  4.01303796600683693   2.02124747019491213 -1.92567478226486166

# Simple logistic regression did better probably because it better reflects
# the true data generating process

# So network accuracy should be between 0.6353 and 0.9741
# and log loss should be between 0.08585132 and 0.716812
# It may exceed these eventually