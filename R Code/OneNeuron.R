# Only source data preparation script if data are missing
if(!("all.dat" %in% ls())) source("./DataPrep.R")

# Let's construct a single neuron that uses the sigmoid function, this should 
# create similar results to the logistic regression in "OtherApproaches.R"

# Need:
# 1) Input initial weights
# 2) Bias with initial weights
# 3) Sigmoid function and its derivatives wrt. weights
# 4) Loop for forward and backward passes

# Evaluator function to convert strings into actual functions/expressions
# Important later
evaluator <- function(cust_str) eval(parse(text = cust_str))

# Initialize input weights for algorithm and symbolic differentiation
outcome_pos <- ncol(all.dat$training)

weights <- rnorm(ncol(all.dat$training))
names(weights) <- c("bias",colnames(all.dat$training[,-outcome_pos]))
names(weights) <- unname(sapply(names(weights), function(name) 
                                      paste0(name,"_weight"))
                         ) 

# Algorithm begins
# Picking learning rate
# Smaller --> Learns more slowly, may take a long time
# Larger --> May learn to quickly and be unstable
learning_rate <- 1/1e1

iter <- 1e4

for(i in 1:iter) {
  # Sampling data frame row, bias is like an input with a constant value of 1
  # Inputs are fixed constants
  inputs <- data.frame(bias = 1,
    all.dat$training[sample(nrow(all.dat$training), 1),])
  
  output <- inputs[,outcome_pos+1]
  inputs <- inputs[,-(outcome_pos+1)]
  
  
  # Creating custom sigmoid function
  # First creating product between inputs and weights
  sum_prod <- paste0(names(weights),"*",inputs)
  
  # Second, collapsing into a string of sums
  sum_prod <- paste0(sum_prod, collapse = " + ")
  
  # Creating actual output sigmoid function for particular weights
  sig_args <- paste0(paste0(names(weights), " = ", weights), collapse = ", ")
  
  # Looks messy, but all this does is create a function that produces the output
  evaluator(
    paste0("sig_out <<- ","function(",sig_args,")", " 1/(1+exp(1)^-(",sum_prod,"))")
  )
  
  # Function that holds partial derivatives
  # Contains both the current output of the function and the partial derivs
  sig_func <- paste0("1/(1+exp(1)^-(",sum_prod,"))")

  sig_der <- evaluator(
    paste0("deriv(output ~ ",
           sig_func,
           ", names(weights)",", funct = T)")
  )
  
  pdevs <- do.call(sig_der, as.list(weights))
  
  # Computing error and applying change to weights using learning rate
  pull <- output-head(pdevs)

  # Gradient inverse blows up when the pull is very large or very small
  grad_inv <- 1/(as.vector(attr(pdevs,"gradient")))
  
  # Ceiling and floor effect for gradient behavior - don't break under
  # extremely large or small pulls
  grad_inv <- sapply(grad_inv, function(grad) {
    # Large values
    if(grad > 1e1) return(1e1)
    if(grad < -1e1) return(-1e1)
    
    # Small values
    if(grad > 0 & grad < 1/1e2) return(1/1e2)
    if(grad < 0 & grad > -1/1e2) return(-1/1e2)
    
    # Small pull leads to huge change in weights? Look into this ###
    if(pull < 1/1e2) return(0)
    
    return(grad)
  })
  
  # Small pull leads to huge change in weights? Look into this ###
  weight_change <- pull * grad_inv * learning_rate
  
  weights <- weights + weight_change
  
  #weights <- weights+(as.vector(attr(pdevs,"gradient"))*learning_rate)

  # Printing out progress, makes everything slow
  #if(any(i/iter == seq(from = 0.1, to = 1, by = 0.1))) 
    #print(paste0("Progress: ",i/iter*100,"%"))
}

## Testing log loss
pred_gen <- function(inputs_df) {
  
  dat <- data.frame(bias = 1,
                    inputs_df)
  
  weighted_inputs <- weights*dat[,-outcome_pos]

  summed_inputs <- apply(weighted_inputs, 1, sum)
  
  1/(1+exp(1)^-summed_inputs)  
}

preds <- pred_gen(all.dat$validation)

lib_load("MLmetrics")

# Log loss
# 1.379183 #
LogLoss(y_pred = preds, y_true = all.dat$validation$Occupancy)

# Accuracy
lib_load("caret")

# 0.561 #
confusionMatrix(data = round(preds,0),
                reference = all.dat$validation$Occupancy)

#x = sum(weights*inputs)

#-(0*log(1/(1+exp(1)^-(x)))+(1-0)*log(1-1/(1+exp(1)^-(x))))
