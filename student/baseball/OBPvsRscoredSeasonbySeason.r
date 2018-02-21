#This program runs a linear regression for runs and (almost) on-base percentage
#for each season. Then the trend of the coefficient is plotted

install.packages("Lahman") #installs the Lahman baseball database
library(Lahman)
Teams$OBP<-(Teams$H+Teams$BB)/(Teams$AB+Teams$BB) #data for SF and HBP are incomplete in this database for some years
counter=1 # intializes a counter variable for working through the dataframe below
df <- data.frame(season=numeric(), #creates an empty dataframe for storing seasonal data
                 relation=numeric(), # calculated linear regression coefficient
                 stderror=numeric(), # calculted standard error
                 minRstandard=numeric(), #finds the lowest z score for an individual team
                 maxRstandard=numeric(), # finds the highest z score for an individual team
                 stringsAsFactors=FALSE)
for (year in c(1903:2016)){
  newTeams <- subset(Teams, yearID==year, select=c(yearID, teamID, OBP, R)) # slices out a subset from Teams
  testmod<-lm(newTeams$R~newTeams$OBP) # does the regression
  se<-sqrt(diag(vcov(testmod)))  # calculates standard error

  df[counter,]<-c(season=year,relation=coef(testmod)[2], stderror=se[2],minRstandard=min(rstandard(testmod)),maxRstandard=max(rstandard(testmod)))
     #sends theseason's linear regression coefficent to the dataframe
  counter<-counter+1
}
dev.new()
plot(df$stderror~df$season)
dev.new()
plot(df$relation~df$season)  # plot to see trends in the coefficient

#A glance at the plots shows there's no trend in either the relationship between OBP and runs scored
#or the standard error over time
