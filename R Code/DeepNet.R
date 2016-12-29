# Only source data preparation script if data are missing
if(!("all.dat" %in% ls())) source("./DataPrep.R")

# Creating a deep network:
# 1) Specifiy number of layers and neurons per layer,
# minimum number of connections
# 2) Expand list of neurons and randomly connect them
# 3) Fill in lists with functions and weights
# 4) Roll everything up in sigmoid
# 5) Differentiate and compile
# 6) Train and test

# Initializing parameters and empty network
layers <- 3
neurons <- 5
min_conn <- 5
sigmoid_const <- -0.2


net <- vector(mode = 'list', length = layers+1)

# Setting up input layer
net[[1]] <- colnames(all.dat$training)[1:ncol(all.dat$training)-1]
names(net[[1]]) <- net[[1]]

# Layer first (i), then neuron number in layer (j)
# Naming network and sampling to find number of connections
for(i in 2:(layers+1)) {
  
  for(j in 1:neurons) {
    
    if(min_conn == neurons) { net[[i]][j] <- min_conn }
      else
    { net[[i]][j] <- sample(x = min_conn:neurons, size = 1, replace = F) }
    
    names(net[[i]])[j] <- paste0("n_",i,"_",j) 
    
  }
}

# Filling in layers with previous layers and adding weights
for(i in 2:(layers+1)) {
  
  for(j in 1:neurons) {
  
    # Sampling previous layer under constraints specified above
    inputs <- sample(x = names(net[[i-1]]),
                     size = net[[i]][j],
                     replace = F)
    
    # Appending weights to layer's inputs
    w_inputs <- sapply(inputs, function(input) {
                                  paste0(input,
                                         " * ",
                                         paste0("w_",i,j,"_",input))
                               })
    
    # Inserting into network
    net[[i]][j] <- paste0(w_inputs, collapse = " + ")
    
    # Adding bias
    net[[i]][j] <- paste0(net[[i]][j], " + ",
                          paste0("bias * w_",i,j,"_bias"))
    
    # Closing with parens
    net[[i]][j] <- paste0("( ",net[[i]][j]," )")
  }
}

# Wrapping functions around nodes
for(i in 2:(layers+1)) {
  
  for(j in 1:neurons) {
  
    net[[i]][j] <- paste0("(log(1+exp(",net[[i]][j],")))")
  
  }
}

# Substitute bottom into layers into next layers
for(i in 3:(layers+1)) {
  # For each neuron
  for(j in 1:neurons) {
    
    # Replace each known node with its underlying function
    for(q in 1:neurons) {
      # Look for the neuron with spaces on either side
      neuron_string <- paste0(" ",names(net[[i-1]][q])," ")
      
      net[[i]][j] <- gsub(pattern = neuron_string,
                          replacement = net[[i-1]][q],
                          x = net[[i]][j])
    }
  }
  
  # Clear out previous layer to save memory
  net[[i-1]] <- ""
}

# Roll into giant final function and into sigmoid, cleaing large objects
# out of RAM to save space
sig_input <- paste0("(",
                    paste0(net[[layers+1]], collapse = " + "),
                    ")"
                    )
rm(net)

sig_fun <- paste0("1/(1+exp(",sigmoid_const,"*(",sig_input,")))")

rm(sig_input)

# Extracting all of the unqiue weights
split_fun <- strsplit(sig_fun, split = " ")[[1]]

weights <- split_fun[sapply(split_fun, grepl, pattern = "[w_][0-9][0-9][_]")]

weights <- unique(weights)

weights <- sort(weights)

# Setup for partial derivative function
outcome_loc <- ncol(all.dat$training)

input_names <- names(all.dat$training)[-outcome_loc]

# Finding which inputs are actually used
input_inx <- which(input_names %in% split_fun)

input_names <- input_names[input_inx]

all_der_names <- c(input_names,
                   "bias",
                   weights)

sig_deriv <- deriv(parse(text=sig_fun),
                   namevec = all_der_names, 
                   func = T)

# Compiling for speed and clearning memory to save space
lib_load("compiler")

sig_deriv <- cmpfun(sig_deriv)

rm(list = setdiff(ls(),
                  c("all.dat","weights","sig_deriv",
                    "outcome_loc","lib_load","input_inx")))

# Intialize weights
w_names <- weights

weights <- rnorm(n = length(weights), mean = 0, sd = 0.1)

names(weights) <- w_names

rm(w_names)

# Separating inputs and outputs
inputs <- all.dat$training[,-outcome_loc]
outputs <- all.dat$training[,outcome_loc]

# Regularization and learning for network
learning_rate <- 1/1e3

grad_reg <- 1/1e2

iter <- 1e5

# Creating sample index vector outside loop for speed and setting up
# default bias weight
samp_ind <- sample(1:nrow(all.dat$training), size = iter, replace = T)

bias <- 1
names(bias) <- "bias"

# Loop begins 
for(i in 1:iter) {
 
  # Compute partial derivatives of sigmoid function given weights and 
  # holding the inputs constant, compact form for speed
  pdevs <- do.call(sig_deriv, as.list(c(inputs[samp_ind[i],input_inx],
                                        bias,
                                        weights))) 
  
  # Getting rid of partial derivatives with respect to inputs and bias const.
  attr(pdevs,"gradient") <- attr(pdevs,"gradient")[-(1:(length(input_inx)+1))]
  
  # Observed             # Predicted output at current input levels
  pull <- outputs[samp_ind[i]] - head(pdevs)#; print(pull)  
  
  # Taking the inverse of the partial derivatives to solve for change in weight
  grad_inv <- 1/attr(pdevs, 'gradient')
  
  # Apply gradient shrinkage, set a ceiling and floor
  grad_inv <- sapply(grad_inv, function(grad) {
    
    # Dealing with NA
    if(is.na(grad)) return(0)
    
    # Regularizes gradient
    if(grad > grad_reg) return(grad_reg)
    if(grad < -grad_reg) return(-grad_reg)
    
    return(grad)
  })

  # Apply learning rate and change weights according to capped partial derivs
  weights <- weights + (pull * grad_inv * learning_rate)
  
  # If weights are basically the same, reset closest ones --needs work
#   nil <- sapply(weights, function(weight) {
#   
#    # Return absolute percentage differences    
#    perc_dif <- abs((abs(weights) - abs(weight))/weights)
#    
#    # Find which weights are too close to this one
#    change_inx <- which(0 < perc_dif & perc_dif < 0.01)
#    
#    # Change the weights that are too close
#    weights[change_inx] <<- rnorm(n = length(change_inx)) + weights[change_inx]
#    
#    return(NULL)
#   #})
#   
}

# Pushing out predictions
pred_gen <- function(inputs_df) {
  
  preds <- vector("numeric", length = nrow(inputs_df))
  
  dat <- data.frame(inputs_df[,-outcome_loc],
                    bias = 1)
  
  for(i in 1:length(preds)) {
    
    input <- as.list(c(dat[i,c(input_inx,ncol(dat))],
                       weights))
    
    preds[i] <- head(do.call(sig_deriv, input))
  }
  
  return(preds)
}

preds <- pred_gen(all.dat$validation)

lib_load("MLmetrics")

# Log loss
# 0.6931471 #
LogLoss(y_pred = preds, y_true = all.dat$validation$Occupancy)

plot(weights)

plot(density(preds))

# Accuracy
#lib_load("caret")

# 0.9707 # Looks like this is peak accuracy
#round(confusionMatrix(data = round(preds,0),
                      #reference = all.dat$validation$Occupancy)$overall,4)
