# So I've learned from things from the old single neuron implementation
# Here are the things that are fixed with this iteration:
# 1) Faster loops, took partial derivative function out of loop & compiled
# 2) Adjust for wacky ceiling and floor effects for sigmoid function pdevs
# 3) Play with learning rate to get better accuracy

# Only source data preparation script if data are missing
if(!("all.dat" %in% ls())) source("./DataPrep.R")

# Initialize input weights for algorithm and symbolic differentiation
outcome_pos <- ncol(all.dat$training)

weights <- rnorm(ncol(all.dat$training))

names(weights) <- c("bias",colnames(all.dat$training[,-outcome_pos]))

names(weights) <- unname(sapply(names(weights), function(name) 
  paste0(name,"_weight"))
) 

# Set up sigmoid function with rate of change parameter 
sum_prod <- paste0(names(weights),"*", colnames(all.dat$training[,-outcome_pos]))

sum_prod <- paste(sum_prod, collapse = " + ")

# Larger parameter means the sigmoid function changes faster
# Treat this as a "random" number between x and y when building
# the larger random network, 0.5 and 1 are decent starting points
roc_sig <- exp(1)/pi

sig_func <- paste0("1/(1+exp(1)^-(",roc_sig,"*","(",sum_prod,")))")

# Partial derivative with respect to inputs (which don't change)
# and weights (which do change), compiling for extra speed and clean up
sig_der <- deriv(parse(text = sig_func),
                 namevec = c(names(weights),
                             colnames(all.dat$training[,-outcome_pos])),
                 funct = T)

lib_load("compiler")

sig_der <- cmpfun(sig_der)

rm(sig_func, sum_prod)

# Separating inputs and outputs
inputs <- all.dat$training[,-outcome_pos]
outputs <- all.dat$training[,outcome_pos]

# Loop setup: regularization for weight adjustment, ceiling/floor caps for 
# partial derivatives, and number of learning iterations
# Max change for any weight in each iteration = learning_rate*grad_reg*pull
# Since pull can be no more than 1, this reduces to = learning_rate*grad_reg
learning_rate <- 1/1e3

grad_reg <- 1/1e1

iter <- 2e6

# Creating sample index vector outside loop for speed and setting up
# default bias weight
samp_ind <- sample(1:nrow(all.dat$training), size = iter, replace = T)

bias <- 1

for(i in 1:iter) {

  # Compute partial derivatives of sigmoid function given weights and 
  # holding the inputs constant, compact form for speed
  pdevs <- do.call(sig_der, as.list(c(weights,
                                      inputs[samp_ind[i],])))
  
  attr(pdevs,"gradient") <- attr(pdevs,"gradient")[1:outcome_pos]
  
          # Observed             # Predicted output at current input levels
  pull <- outputs[samp_ind[i]] - head(pdevs)
  
  # Taking the inverse of the partial derivatives to solve for change in weight
  grad_inv <- 1/attr(pdevs, 'gradient')
 
  # Apply gradient shrinkage, set a ceiling and floor
  grad_inv <- sapply(grad_inv, function(grad) {
    
    # Regularizes gradient
    if(grad > grad_reg) return(grad_reg)
    if(grad < -grad_reg) return(-grad_reg)
    
    return(grad)
  })
  
  # Apply learning rate and change weights according to capped partial derivs
  weights <- weights + (pull * grad_inv * learning_rate)
}

# To compare to logistic regression, doesn't appear to be using the exact
# functional form of the logit function but they're somewhat close
# Sigmoid is a special case of the logistic
# http://stats.stackexchange.com/questions/204484/what-are-the-differences-between-logistic-function-and-sigmoid-function
# weights <- c(-4.33202811558247181, -0.91922576450224724, 1.71425273557116387, 4.01303796600683693, 2.02124747019491213, -1.92567478226486166)


# Prediction generation and testing log loss
pred_gen <- function(inputs_df) {
  
  dat <- data.frame(bias = 1,
                    inputs_df)
  
  for(i in 1:nrow(dat)) {
    
    dat[i,] <- dat[i,]*weights
    
  }

  summed_inputs <- rowSums(dat)
  
  1/(1+exp(1)^-(summed_inputs))  
}

preds <- pred_gen(all.dat$validation)

lib_load("MLmetrics")

# Log loss
# 0.1456651 #
LogLoss(y_pred = preds, y_true = all.dat$validation$Occupancy)

# Accuracy
lib_load("caret")

# 0.9786 # Looks like this is peak accuracy
round(confusionMatrix(data = round(preds,0),
                      reference = all.dat$validation$Occupancy)$overall,4)
