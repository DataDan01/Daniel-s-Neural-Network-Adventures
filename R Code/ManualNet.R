# Only source data preparation script if data are missing
if(!("all.dat" %in% ls())) source("./DataPrep.R")

# This is my first attempt at building a mulit-layer network manually by hand
# Here's the top level map:
# Inputs --> "Randomly" connected softplus/minus functions --> Sigmoid --> Pred
# What does a custom softplus function look like?

cust_softplus <- function(const, weighted_inputs) {
 
   const*log((1+exp(1)^weighted_inputs))
  
}

lib_load("manipulate")

# Gets weird when the constant is zero but otherwise it essentially
# constrains outputs to be positive or negative in a smooth fashion
manipulate(
  plot(cust_softplus(const, weighted_inputs = seq(from = -3, to = 3, by = 0.01)),
       x = seq(from = -3, to = 3, by = 0.01),
       main = "Soft Plus/Minus Function",
       xlab = "Weighted Inputs",
       ylab = "Output",
       type = "l"),
  const = slider(min = -2, max = 2, step = 0.1)
)

# When looking at functional form, how many softplus/softminus neurons
# can we have that take in between 1 and all of the inputs?
# It should be = (power set - one empty set)
(2^(ncol(all.dat$training)-1)-1) # (2^5-1)

# First attempt at a plan to generate network structure:
# 1) Pick number of nodes in hidden layer
# 2) Create random connections between inputs and hidden layer
# 3) Generate and capture functional forms for hidden layer neurons
# 4) Roll everything up into the final sigmoid function
# 5) Compute partial derivatives with respect to everything

# 1) Pick number of nodes in hidden layer, 6 to start to see how things go
# +
# 2) Create random connections between inputs and hidden layer
layer <- 1

hidden_size <- 6

# Making layer larger than needed and then reducing to unique input
# combinations, random sampling of inputs can produce duplicate nodes
hidden_nodes <- vector("list", hidden_size*2)

input_names <- colnames(all.dat$training)[-ncol(all.dat$training)]

for(i in 1:length(hidden_nodes)) {
  
  hidden_nodes[[i]] <- sample(input_names,
                              size = sample(1:length(input_names),1))
  
  hidden_nodes[[i]] <- sapply(hidden_nodes[[i]], function(string) 
    paste0(string,"_w"))
}

hidden_nodes <- lapply(hidden_nodes, sort)

hidden_nodes <- unique(hidden_nodes)[1:hidden_size]

# Naming nodes
for(i in 1:hidden_size) {
  
  names(hidden_nodes)[i] <- paste0("n_",layer,i) 
  
}

# Adding in biases
for(i in 1:hidden_size) {
  
  len <- length(hidden_nodes[[i]])
  
  hidden_nodes[[i]][len+1] <- paste0("bias",layer,i,"_w")
  
  names(hidden_nodes[[i]])[len+1] <- "bias"
  
}

# 3) Generate and capture functional forms for hidden layer neurons
hidden_functs <- hidden_nodes

for(i in 1:hidden_size) {
  
  string_vect <- hidden_nodes[[i]]
  
  # Computing a string of sum(weights*inputs)
  prods <- paste0("(",string_vect,"*",names(string_vect),")")
  
  sums <- paste0(prods,collapse = "+")
  
  funct <- paste0("log(1+exp(1)^(",sums,"))")
  
  hidden_functs[[i]] <- funct
  
}

# 4) Roll everything up into the final sigmoid function
output_node <- vector("list", length = hidden_size)

# Creating weight list for output node
for(i in 1:length(output_node)) {
  
  names(output_node)[i] <- paste0("o_",i)
  
  output_node[i] <- paste0(names(hidden_nodes)[i],"_w")

}

# Creating final sigmoid function
sig_func <- vector("character", length = hidden_size)

names(sig_func) <- names(output_node)

for(i in 1:length(sig_func)) {
  
  sig_func[i] <- paste0("(",output_node[i],"*",hidden_functs[i],")")
  
}

sig_func <- paste0("(",paste0(sig_func, collapse = "+"),")")

# Setting up constant for sigmoid steepness
sig_const <- exp(1)/pi

sig_func <- paste0("1/(1+exp(1)^-(",sig_const,"*(",sig_func,")))")

# 5) Compute partial derivatives with respect to everything
all_der_names <- c(input_names,
                   "bias",
                   unname(unlist(hidden_nodes)),
                   unname(unlist(output_node)))

all_der_names <- unique(all_der_names)

sig_der <- deriv(parse(text=sig_func),
                 namevec = all_der_names, 
                 func = T)

# Compiling for extra speed
lib_load("compiler")

sig_der <- cmpfun(sig_der)

# Cleaing up
zzz_need <- which(ls() %in% c("sig_der","all_der_names","all.dat","lib_load"))

rm(list=ls()[-zzz_need])

# Training setup begins
# Initialize random weights for hidden and output layers
outcome_pos <- ncol(all.dat$training)
no_inputs <- outcome_pos-1

weights <- rnorm(length(all_der_names)-(no_inputs+1))
names(weights) <- all_der_names[-(1:(no_inputs+1))]


# Separating inputs and outputs
inputs <- all.dat$training[,-outcome_pos]
outputs <- all.dat$training[,outcome_pos]

## Return all below to train more
# Regularization and learning for network
learning_rate <- 1/1e2

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
  pdevs <- do.call(sig_der, as.list(c(weights,
                                      bias,
                                      inputs[samp_ind[i],])))
  
  attr(pdevs,"gradient") <- attr(pdevs,"gradient")[-(1:outcome_pos)]
  
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

# Pushing out predictions
pred_gen <- function(inputs_df) {
  
  preds <- vector("numeric", length = nrow(inputs_df))
  
  dat <- data.frame(bias = 1,
                    inputs_df[,-outcome_pos])
  
  for(i in 1:length(preds)) {
    
    preds[i] <- head(do.call(sig_der, as.list(c(weights,
                                                dat[i,]))))
  }
  
  return(preds)
}

preds <- pred_gen(all.dat$validation)

lib_load("MLmetrics")

# Log loss
# 0.09883161 #
LogLoss(y_pred = preds, y_true = all.dat$validation$Occupancy)

# Accuracy
lib_load("caret")

# 0.9707 # Looks like this is peak accuracy
round(confusionMatrix(data = round(preds,0),
                      reference = all.dat$validation$Occupancy)$overall,4)

