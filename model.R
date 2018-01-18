csv.load <- function(file_name, features_name){
  data = read.csv(file_name)
  data = data[features_name]
  return(data)
}

csv.save <- function(data){
  df = data.frame(data$PassengerId, data$Survived)
  colnames(df) <- c("PassengerId", "Survived")
  write.table(df, 'out.csv', sep = ',', row.names = FALSE)
}

sex_to_numeric <- function(X){
  if (is.na(X))
    return (NA)
  else if (X == "male")
    return (1)
  else if (X == "female")
    return (0)
  else
    return (NA)
}

fill_na <- function(X, value){
  if (is.na(X))
    return (value)
  else
    return (X)
}

# 1) Load dataset from csv
# 2) Extract relevant information
data = csv.load("data/train.csv", c("Survived", "Pclass", "Sex", "Age", "Fare"))

# 3) Transform non numeric features into numeric features
data$Sex = sapply(data$Sex, sex_to_numeric)

# 4) Replace NA value by the feature's mean
data$Pclass <- sapply(data$Pclass, fill_na, mean(na.omit(data$Pclass)))
data$Sex    <- sapply(data$Sex,    fill_na, mean(na.omit(data$Sex)))
data$Age    <- sapply(data$Age,    fill_na, mean(na.omit(data$Age)))
data$Fare   <- sapply(data$Fare,   fill_na, mean(na.omit(data$Fare)))

# 5) Process logistic regression
result = glm(Survived ~ ., data, family=binomial(logit))

# 6) For each prediction if it is superior to 0.5 then predict 1 else predict 0
result$fitted = as.integer(result$fitted > 0.5)

accuracy = as.integer(result$fitted == data$Survived)

print("Accuracy in training set : ")
print(mean(accuracy))


# 1) Load dataset from csv
# 2) Extract relevant information
data = csv.load("data/test.csv", c("PassengerId", "Pclass", "Sex", "Age", "Fare"))

# 3) Transform non numeric features into numeric features
data$Sex = sapply(data$Sex, sex_to_numeric)

# 4) Replace NA value by the feature's mean
data$Pclass <- sapply(data$Pclass, fill_na, mean(na.omit(data$Pclass)))
data$Sex    <- sapply(data$Sex,    fill_na, mean(na.omit(data$Sex)))
data$Age    <- sapply(data$Age,    fill_na, mean(na.omit(data$Age)))
data$Fare   <- sapply(data$Fare,   fill_na, mean(na.omit(data$Fare)))

# 5) Process logistic prediction
result_testset = predict(result, newdata=data)

# 6) For each prediction if it is superior to 0.5 then predict 1 else predict 0
data$Survived = as.integer(result_testset > 0.5)

csv.save(data)
