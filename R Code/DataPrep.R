## This script prepares the data for use in the network

# This function makes sure the appropriate libraries are installed
# and then loads them
# It will be used throughout this project
lib_load <- function(library_name) {
  
  if(!require(library_name, character.only = T, quietly = T)) 
      install.packages(library_name)

  require(library_name, character.only = T, quietly = T)
}

# Download, unzip, and load in the data
# Credit goes to: Accurate occupancy detection of an office room from light, temperature, humidity and CO2 measurements using statistical learning models. Luis M. Candanedo, VÃ©ronique Feldheim. Energy and Buildings. Volume 112, 15 January 2016, Pages 28-39.
# This is a data set that contains information about rooms that is 
# used to predict occupancy, the features are:
    # date time year-month-day hour:minute:second 
    # Temperature, in Celsius 
    # Relative Humidity, % 
    # Light, in Lux 
    # CO2, in ppm 
    # Humidity Ratio, Derived quantity from temperature and relative humidity, in kgwater-vapor/kg-air 
    # Occupancy, 0 or 1, 0 for not occupied, 1 for occupied status

# Only download and extract the data if it's not already there
if("./dat.zip" %in% list.files("./data")) {

  download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip",
                "./data/dat.zip")
  
  unzip(zipfile = "./data/dat.zip", exdir = "./data")
}

all.dat <- list(
    training = read.csv("./data/datatraining.txt"),
    validation = read.csv("./data/datatest.txt"),
    test = read.csv("./data/datatest2.txt")
)

# Recording the training mean and SDs for the variables
# Will be used to consistently preprocess data
train_means <- sapply(all.dat$training[,-c(1,7)], mean)
train_sds <- sapply(all.dat$training[,-c(1,7)], sd)

# Create a single function to clean up the data set
cleaner <- function(df) {
  
  # Remove time stamp
  new_df <- df[,-1]
  
  # Normalize all data according to training SD and Mean!
  # Important to keep preprocess consistent across data
  # Should generally avoid loops but this one is quick
  for(i in 1:(ncol(new_df)-1)) {
    
    new_df[,i] <- (new_df[,i]-train_means[i])/train_sds[i]
    
  }
  
  return(new_df)
}

# Normalizing all data and cleaning up
all.dat <- lapply(all.dat, cleaner) 

rm(train_means, train_sds, cleaner)

# Looks like these data shift between samples
# apply(all.dat$validation, 2, mean)
# apply(all.dat$validation, 2, sd)

# apply(all.dat$test, 2, mean)
# apply(all.dat$test, 2, sd)


