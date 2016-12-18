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
neurons <- 3
min_conn <- 2

net <- vector(mode = 'list', length = layers+1)

# Setting up input layer
net[[1]] <- colnames(all.dat$training)[1:ncol(all.dat$training)-1]
names(net[[1]]) <- net[[1]]

# Layer first (i), then neuron number in layer (j)
# Naming network and sampling to find number of connections
for(i in 2:(layers+1)) {
  
  for(j in 1:neurons) {
    
    net[[i]][j] <- sample(min_conn:neurons, 1)
    
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
                                         paste0("w(",i,j,")_",input))
                               })
    
    # Inserting into network
    net[[i]][j] <- paste0(w_inputs, collapse = " + ")
    
    net[[i]][j] <- paste0("(",net[[i]][j],")")
  }
}

# Wrapping functions around nodes
for(i in 2:(layers+1)) {
  
  for(j in 1:neurons) {
  
    net[[i]][j] <- paste0("(log(1+exp(1)^",net[[i]][j],"))")
  
  }
}

# Substitute bottom into layers into next layers
for(i in 3:(layers+1)) {
  
  for(j in 1:neurons) {
    
    # Find each node and replace its name with the underlying function
    # For nodes in the layer above
    
  }
}

## Fix bias, need to add to every node ##