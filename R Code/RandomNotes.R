# These are random pieces of code, most of these don't work
# This is just a place for me to play with ideas
sig_out <- evaluator(
  paste0("function(",
         paste0(names(weights),collapse = ", "), ")",
         " { ",sum_prod," }")
)

sig_out(weights)

# Function that holds partial derivatives
sig_der <- evaluator(
  paste0("deriv(pred ~ ",sum_prod,", names(weights)",", funct = T)")
)

sig_out <- evaluator(
  paste0("function(",
         paste0(names(weights),collapse = ", "), ")",
         " { ",sum_prod," }")
)
class(sig_der())

eval(parse(text=
             paste0("deriv(",sig_funct,", names(weights)",", funct = T)")
))


sum_prod_str <- lapply(1:length(weights), function(index) {
  
  paste0(names(inputs)[index],"*",
         paste0("weight_"))
         
})
  
  
  sigfun <- 1/(1+exp(1)^-(x))
  
  der <- deriv((x),
               c("bias",colnames(all.dat$training)))
  
  der <- deriv((y ~ sin(cos(x) * y)), c("x","y"), func = TRUE)
  
  der(2,3)
  
  
  # Testing single neuron loop
  head(pdevs)
  output
  head(do.call(sig_der, as.list(new_weights)))
  
  abs(output-head(pdevs)) > abs(output-head(do.call(sig_der, as.list(new_weights))))
  
  # Attempt at training single neuron in parallel, doesn't work because
  # nodes don't have access to parent evnior
  
  # Training in parallel
  lib_load("parallel")
  
  cl <- makeCluster(detectCores()-2)
  
  clusterExport(cl, list("weight_train_cmp",
                         "evaluator",
                         "weights",
                         "outcome_pos",
                         "learning_rate",
                         "all.dat"))
  
  weights <- unlist(tail(parLapply(cl, 1:5e4, weight_train_cmp), 1))
  
  # Not worth compiling the function and lapplying it because it holds all
  # outputs at once, thus slowing everything down
  # Loop is the fastest current approach
  # Compiling the function to get a bit more speed
  lib_load("compiler")
  
  weight_train_cmp <- cmpfun(weight_train)
  
  weights <- unlist(tail(lapply(1:5e3, weight_train_cmp), 1))
  
  weights
  
  # Was inside function for lapply and compilation
  assign("weights",
         weights+(as.vector(attr(pdevs,"gradient"))*pull*learning_rate),
         envir = globalenv())
  
  # Sigmoid derivative
  
  sig_der <- evaluator(
    paste0("deriv(output ~ ",
           paste0("1/(1+exp(1)^-(",sum_prod,"))"),
           ", names(weights)",", funct = T)")
  )
  

  
  # Cross entropy cost function
  # Connect learning rate of pull specification with
  # cost function result?
  cost_func <- paste0("-(",output,"*","log(",sig_func,"))","+",
                      "(1-",output,")","*","log(1-",sig_func,")))")
  
  cost_der <- evaluator(
    paste0("deriv(cost ~ ", cost_func,", names(weights)",", funct = T)")
  )