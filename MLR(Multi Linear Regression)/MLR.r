##############################Problem 1#################################
# Load the Cars dataset
library(readr)
startup_data <- read.csv("C:\\Users\\hp\\Desktop\\MLR assi\\50_Startups.csv", header = TRUE )
View(startup_data)

attach(startup_data)

# Normal distribution
qqnorm(R.D.Spend)
qqline(R.D.Spend)

# Exploratory data analysis:
summary(startup_data)

#label encoding
factors <-  as.factor(startup_data$State)
startup_data$State <- as.numeric(factors)

# Scatter plot
plot(R.D.Spend, Administration) # Plot relation ships between each X with Y
plot(Marketing.Spend, Profit)

# Or make a combined plot
pairs(startup_data)   # Scatter plot for all pairs of variables
plot(startup_data)

cor(R.D.Spend, Profit)
cor(startup_data) # correlation matrix

# The Linear Model of interest
model.startup <- lm(Profit ~ R.D.Spend + Administration + Marketing.Spend + State, data = startup_data) # lm(Y ~ X)
summary(model.startup)

model.startupRD <- lm(Profit ~ R.D.Spend)
summary(model.startupRD)

model.startupA <- lm(Profit ~ Administration)
summary(model.startupA)

model.startupm <- lm(Profit ~ Marketing.Spend)
summary(model.startupm)

model.startupS <- lm(Profit ~ State)
summary(model.startupS)

#Scatter plot matrix with Correlations inserted in graph
install.packages("GGally")
library(GGally)
ggpairs(startup_data)

### Partial Correlation matrix
install.packages("corpcor")
library(corpcor)
cor(startup_data)

cor2pcor(cor(startup_data))

# Diagnostic Plots
install.packages("car")
library(car)

plot(model.startup)# Residual Plots, QQ-Plot, Std. Residuals vs Fitted, Cook's distance

qqPlot(model.startup, id.n = 5) # QQ plots of Standardized residuals, helps identify outliers

# Deletion Diagnostics for identifying influential observations
influenceIndexPlot(model.startup, id.n = 3) # Index Plots of the influence measures
influencePlot(model.startup, id.n = 3) # A user friendly representation of the above

# Regression after deleting the 77th observation
model.startup1 <- lm(Profit ~ R.D.Spend + Administration + Marketing.Spend + State, data = startup_data[-46, ])
summary(model.startup1)

### Variance Inflation Factors
vif(model.startup)  # VIF is > 10 => col-linearity

# Regression model to check R^2 on Independent variables
VIFR <- lm(R.D.Spend ~ Administration + Marketing.Spend + State , data = startup_data)
VIFA <- lm(Administration ~ R.D.Spend + Marketing.Spend + State, data = startup_data)
VIFM <- lm(Marketing.Spend ~ R.D.Spend + Administration + State, data = startup_data)
VIFS <- lm(State ~ R.D.Spend + Administration + Marketing.Spend , data = startup_data)

summary(VIFR)
summary(VIFA)
summary(VIFM)
summary(VIFS)

# VIF of R.D.Spend
1/(1-0.597) 

#Added Variable Plots
avPlots(model.startup, id.n = 2, id.cex = 0.8, col = "red")

# Added Variable Plots
avPlots(model.startup1, id.n = 2, id.cex = 0.8, col = "red")

# Variance Influence Plot
vif(model.startup1)

# Evaluation Model Assumptions
plot(model.startup1)
plot(model.startup1$fitted.values, model.final1$residuals)

qqnorm(model.final1$residuals)
qqline(model.final1$residuals)

# Subset selection
install.packages("leaps")
library(leaps)

#Best Subset Selection
lm_best <- regsubsets(Profit ~ ., data = startup_data, nvmax = 15)
summary(lm_best)
summary(lm_best)$adjr2
which.max(summary(lm_best)$adjr2)
coef(lm_best, 3)

#forward subset selection
lm_forward <- regsubsets(Profit ~ ., data = startup_data, nvmax = 15, method = "forward")
summary(lm_forward)

# Data Partitioning
n <- nrow(startup_data)
n1 <- n * 0.7
n2 <- n - n1
samp <- sample(1:n, n1)
train <- startup_data[samp , ]
test <- startup_data[-samp, ]

# Model Training
model <- lm(Profit ~ R.D.Spend + Administration + Marketing.Spend + State, data = train)
summary(model)

#predicting on test data
pred <- predict(model, newdata = test)
actual <- test$Profit
error <- actual - pred

#rmse of error on test data
test.rmse <- sqrt(mean(error**2))
test.rmse

#train rmse
train.rmse <- sqrt(mean(model$residuals**2))
train.rmse

# Step AIC
install.packages("MASS")
library(MASS)
stepAIC(model)

########################################Problem 2#####################################
# Load the Cars dataset
library(readr)
comp_data <- read.csv("C:\\Users\\hp\\Desktop\\MLR assi\\Computer_Data.csv", header = TRUE )
View(comp_data)

attach(comp_data)

# Normal distribution
qqnorm(price)
qqline(price)

# Exploratory data analysis:
summary(comp_data)

#label encoding
fact_cd <-  as.factor(cd)
comp_data$cd <- as.numeric(fact_cd)
fact_multi <- as.factor(multi)
comp_data$multi <- as.numeric(fact_multi)
fact_premium <- as.factor(premium)
comp_data$premium <- as.numeric(fact_premium)

#droping unwanted column
comp_data <- comp_data[,2:11]

# Scatter plot
plot(speed, hd) # Plot relation ships between each X with Y
plot(ram, price)

# Or make a combined plot
pairs(comp_data)   # Scatter plot for all pairs of variables
plot(comp_data)

cor(speed, price)
cor(comp_data) # correlation matrix

# The Linear Model of interest
model.comp <- lm(price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend, data = comp_data) # lm(Y ~ X)
summary(model.comp)

model.compsp <- lm(price ~ speed)
summary(model.compsp)

model.comphd <- lm(price ~ hd)
summary(model.comphd)

model.compram <- lm(price ~ ram)
summary(model.compram)

model.compsc <- lm(price ~ screen)
summary(model.compsc)

model.compcd <- lm(price ~ cd)
summary(model.compcd)

model.compmul <- lm(price ~ multi)
summary(model.compmul)

model.comppr <- lm(price ~ premium)
summary(model.comppr)

model.compads <- lm(price ~ ads)
summary(model.compads)

model.comptr <- lm(price ~ trend)
summary(model.comptr)

#Scatter plot matrix with Correlations inserted in graph
install.packages("GGally")
library(GGally)
ggpairs(comp_data)

### Partial Correlation matrix
install.packages("corpcor")
library(corpcor)
cor(comp_data)

cor2pcor(cor(comp_data))

# Diagnostic Plots
install.packages("car")
library(car)

plot(model.comp)# Residual Plots, QQ-Plot, Std. Residuals vs Fitted, Cook's distance

qqPlot(model.comp, id.n = 5) # QQ plots of Standardized residuals, helps identify outliers

# Deletion Diagnostics for identifying influential observations
influenceIndexPlot(model.comp, id.n = 3) # Index Plots of the influence measures
influencePlot(model.comp, id.n = 3) # A user friendly representation of the above

# Regression after deleting the 4478,3784 observation
model.comp1 <- lm(price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend, data = comp_data[-c(4478,3784), ])
summary(model.comp1)

### Variance Inflation Factors
vif(model.comp)  # VIF is > 10 => col-linearity

#Added Variable Plots for model.comp
avPlots(model.comp, id.n = 2, id.cex = 0.8, col = "red")

# Added Variable Plots for model.comp1
avPlots(model.comp1, id.n = 2, id.cex = 0.8, col = "red")

# Variance Influence Plot
vif(model.startup1)

# Evaluation Model Assumptions
plot(model.comp)
plot(model.comp$fitted.values, model.comp$residuals)

qqnorm(model.comp$residuals)
qqline(model.comp$residuals)

# Subset selection
install.packages("leaps")
library(leaps)

#Best Subset Selection
lm_best <- regsubsets(price ~ ., data = comp_data, nvmax = 15)
summary(lm_best)
summary(lm_best)$adjr2
which.max(summary(lm_best)$adjr2)
coef(lm_best, 3)

#forward subset selection
lm_forward <- regsubsets(price ~ ., data = comp_data, nvmax = 15, method = "forward")
summary(lm_forward)

# Data Partitioning
n <- nrow(comp_data)
n1 <- n * 0.7
n2 <- n - n1
samp <- sample(1:n, n1)
train <- comp_data[samp , ]
test <- comp_data[-samp, ]

# Model Training
model <- lm(price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend, data = train)
summary(model)

#predicting on test data
pred <- predict(model, newdata = test)
actual <- test$price
error <- actual - pred

#rmse of error on test data
test.rmse <- sqrt(mean(error**2))
test.rmse

#train rmse
train.rmse <- sqrt(mean(model$residuals**2))
train.rmse

# Step AIC
install.packages("MASS")
library(MASS)
stepAIC(model)

###########################################Problem 3#########################################
# Load the Cars dataset
library(readr)
car_data <- read.csv("C:\\Users\\hp\\Desktop\\MLR assi\\ToyotaCorolla.csv", header = TRUE )
View(car_data)

attach(car_data)

# Normal distribution
qqnorm(Price)
qqline(Price)

# Exploratory data analysis:
summary(car_data)

#feature selection
cols <- c('Price','Age_08_04', 'KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight')
car_data <- car_data[cols]

# Scatter plot
plot(Price, KM) # Plot relation ships between each X with Y
plot(cc, Weight)

# Or make a combined plot
pairs(car_data)   # Scatter plot for all pairs of variables
plot(car_data)

cor(cc, Weight)
cor(car_data) # correlation matrix

# The Linear Model of interest
model.car <- lm(Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight, data = car_data) # lm(Y ~ X)
summary(model.car)

model.carage <- lm(Price ~ Age_08_04)
summary(model.carage)

model.carKM <- lm(Price ~ KM)
summary(model.carKM)

model.carHP <- lm(Price ~ HP)
summary(model.carHP)

model.carcc <- lm(Price ~ cc)
summary(model.carcc)

model.carDr <- lm(Price ~ Doors)
summary(model.carDr)

model.carGr <- lm(Price ~ Gears)
summary(model.carGr)

model.carQ <- lm(Price ~ Quarterly_Tax)
summary(model.carQ)

model.carW <- lm(Price ~ Weight)
summary(model.carW)

#Scatter plot matrix with Correlations inserted in graph
install.packages("GGally")
library(GGally)
ggpairs(car_data)

### Partial Correlation matrix
install.packages("corpcor")
library(corpcor)
cor(car_data)

cor2pcor(cor(car_data))

# Diagnostic Plots
install.packages("car")
library(car)

plot(model.car)# Residual Plots, QQ-Plot, Std. Residuals vs Fitted, Cook's distance

qqPlot(model.car, id.n = 5) # QQ plots of Standardized residuals, helps identify outliers

# Deletion Diagnostics for identifying influential observations
influenceIndexPlot(model.car, id.n = 3) # Index Plots of the influence measures
influencePlot(model.car, id.n = 3) # A user friendly representation of the above

# Regression after deleting the 81th observation
model.car1 <- lm(Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight, data = car_data[-81, ])
summary(model.car1)

### Variance Inflation Factors
vif(model.car)  # VIF is > 10 => col-linearity

#Added Variable Plots for model.comp
avPlots(model.car, id.n = 2, id.cex = 0.8, col = "red")

# Added Variable Plots for model.comp1
avPlots(model.car1, id.n = 2, id.cex = 0.8, col = "red")

# Variance Influence Plot
vif(model.car1)

#model after removing Doors
final.model <- lm(Price ~ Age_08_04 + KM + HP + cc + Gears + Quarterly_Tax + Weight, data = car_data[-81, ])
summary(final.model)

# Evaluation Model Assumptions
plot(final.model)
plot(final.model$fitted.values, final.model$residuals)

qqnorm(final.model$residuals)
qqline(final.model$residuals)

# Subset selection
install.packages("leaps")
library(leaps)

#Best Subset Selection
lm_best <- regsubsets(Price ~ ., data = car_data, nvmax = 15)
summary(lm_best)
summary(lm_best)$adjr2
which.max(summary(lm_best)$adjr2)
coef(lm_best, 3)

#forward subset selection
lm_forward <- regsubsets(Price ~ ., data = car_data, nvmax = 15, method = "forward")
summary(lm_forward)

# Data Partitioning
n <- nrow(car_data)
n1 <- n * 0.7
n2 <- n - n1
samp <- sample(1:n, n1)
train <- car_data[samp , ]
test <- car_data[-samp, ]

# Model Training without Doors
model <- lm(Price ~ Age_08_04 + KM + HP + cc + Gears + Quarterly_Tax + Weight, data = train)
summary(model)

#predicting on test data
pred <- predict(model, newdata = test)
actual <- test$Price
error <- actual - pred

#rmse of error on test data
test.rmse <- sqrt(mean(error**2))
test.rmse

#train rmse
train.rmse <- sqrt(mean(model$residuals**2))
train.rmse

# Step AIC
install.packages("MASS")
library(MASS)
stepAIC(model)

####################################################problem 4#####################################
# Load the Cars dataset
library(readr)
avacado_data <- read.csv("C:\\Users\\hp\\Desktop\\MLR assi\\Avacado_Price.csv", header = TRUE )
View(avacado_data)

attach(avacado_data)

# Normal distribution
qqnorm(AveragePrice)
qqline(AveragePrice)

# Exploratory data analysis:
summary(avacado_data)

#label encoding
fact_type <-  as.factor(type)
avacado_data$type <- as.numeric(fact_type)


#dropping unwanted column
avacado_data <- avacado_data[,1:10]

# Scatter plot
plot(tot_ava1, tot_ava2) # Plot relation ships between each X with Y
plot(tot_ava3, tot_ava1)

# Or make a combined plot
pairs(avacado_data)   # Scatter plot for all pairs of variables
plot(avacado_data)

cor(AveragePrice, XLarge.Bags)
cor(avacado_data) # correlation matrix

# The Linear Model of interest
model.avacado <- lm(AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge.Bags + type, data = avacado_data) # lm(Y ~ X)
summary(model.avacado)

model.avacadoTV <- lm(AveragePrice ~ Total_Volume)
summary(model.avacadoTV)

model.avacadotv1 <- lm(AveragePrice ~ tot_ava1)
summary(model.avacadotv1)

model.avacadotv2 <- lm(AveragePrice ~ tot_ava2)
summary(model.avacadotv2)

model.avacadotv3 <- lm(AveragePrice ~ tot_ava3)
summary(model.avacadotv3)

model.avacadoTB <- lm(AveragePrice ~ Total_Bags)
summary(model.avacadoTB)

model.avacadoSB <- lm(AveragePrice ~ Small_Bags)
summary(model.avacadoSB)

model.avacadoLB <- lm(AveragePrice ~ Large_Bags)
summary(model.avacadoLB)

model.avacadoXB <- lm(AveragePrice ~ XLarge.Bags)
summary(model.avacadoXB)

model.avacadot <- lm(AveragePrice ~ type)
summary(model.avacadot)

#Scatter plot matrix with Correlations inserted in graph
install.packages("GGally")
library(GGally)
ggpairs(avacado_data)

### Partial Correlation matrix
install.packages("corpcor")
library(corpcor)
cor(avacado_data)

cor2pcor(cor(comp_data))

# Diagnostic Plots
install.packages("car")
library(car)

plot(model.avacado)# Residual Plots, QQ-Plot, Std. Residuals vs Fitted, Cook's distance

qqPlot(model.avacado, id.n = 5) # QQ plots of Standardized residuals, helps identify outliers

# Deletion Diagnostics for identifying influential observations
influenceIndexPlot(model.avacado, id.n = 3) # Index Plots of the influence measures
influencePlot(model.avacado, id.n = 3) # A user friendly representation of the above

# Regression after deleting the 15561,17469 , 5486 ,14126,17429 observation
model.avacado1 <- lm(AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge.Bags + type, data = avacado_data[-c(15561,17469 , 5486 ,14126,17429), ])
summary(model.avacado1)

### Variance Inflation Factors
vif(model.avacado)  # VIF is > 10 => col-linearity

#Added Variable Plots for model.comp
avPlots(model.avacado, id.n = 2, id.cex = 0.8, col = "red")
#Total_Bags , Small_Bags ,Large_Bags, XLarge.Bags are depended on output variable so can be removed

# Added Variable Plots for model.comp1
avPlots(model.avacado1, id.n = 2, id.cex = 0.8, col = "red")

# Variance Influence Plot
vif(model.avacado1)

# Evaluation Model Assumptions
plot(model.avacado)
plot(model.avacado$fitted.values, model.avacado$residuals)

qqnorm(model.avacado$residuals)
qqline(model.avacado$residuals)

# Subset selection
install.packages("leaps")
library(leaps)

#Best Subset Selection
lm_best <- regsubsets(AveragePrice ~ ., data = avacado_data, nvmax = 15)
summary(lm_best)
summary(lm_best)$adjr2
which.max(summary(lm_best)$adjr2)
coef(lm_best, 3)

#forward subset selection
lm_forward <- regsubsets(AveragePrice ~ ., data = avacado_data, nvmax = 15, method = "forward")
summary(lm_forward)

# Data Partitioning
n <- nrow(avacado_data)
n1 <- n * 0.7
n2 <- n - n1
samp <- sample(1:n, n1)
train <- avacado_data[samp , ]
test <- avacado_data[-samp, ]

# Model Training
model <- lm(AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3  + type, data = train)
summary(model)

#predicting on test data
pred <- predict(model, newdata = test)
actual <- test$AveragePrice
error <- actual - pred

#rmse of error on test data
test.rmse <- sqrt(mean(error**2))
test.rmse

#train rmse
train.rmse <- sqrt(mean(model$residuals**2))
train.rmse

# Step AIC
install.packages("MASS")
library(MASS)
stepAIC(model)
############################################################END###################################