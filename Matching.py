import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy.spatial.distance import cdist

# Adding a column to the PSMinput1 file showing which users are in treated group
# PSMinput1 is a csv file contains external variables of each user.
# Here, we have the number of followers, Friends, TweetsPerMonth, FoursquarePerMonth, and Gender
def makeMatchInput(Treated):
    # Tretaed is a file contained users in treated group
    with open(Treated) as f:
        content = f.readlines()
        treated = []
        for x in content:
            treated.append(x.split(',')[0])
        # PSMinput1 is a csv file contains external variables of each user
        with open("PSMinput1.csv") as input1:
            content2 = input1.readlines()
            # PSMinput file is the same as PSMinput1 but contains a new
            # column showing the user is a member of treated group or not
            f = open('PSMinput.csv', 'w')
            # Writing the header o PSMinput1 file to the new PSMinput file
            f.write(content2[0].rstrip())
            f.write('\n')
            for x in range(1, len(content2)):
                a = content2[x].split(',')
                f.write(a[0])
                f.write(',')
                if a[0] in treated:
                    f.write('1')
                else:
                    f.write('0')
                for y in range(2, len(a)):
                    f.write(',')
                    f.write(a[y].rstrip())
                f.write('\n')
            f.close()
# Calculating the probability of every user to be in the
# treated group using logistic regression
def GetPS():
    print("calculating logistic regression")
    # Reading the PSMinput using pandas
    df = pd.read_csv("PSMinput.csv")
    df.columns = ["UserId","Admited","Follower","Friend","TweetsPerMonth","FoursquarePerMonth","Gender"]
    cols_to_keep = ['Admited', 'Follower', 'Friend', "TweetsPerMonth","FoursquarePerMonth", "Gender"]
    data = df[cols_to_keep]
    train_cols = data.columns[1:]
    # Training the model on the data.
    logit = sm.Logit(data['Admited'], data[train_cols])
    result = logit.fit()
    combos = pd.DataFrame(df)
    combos['admit_pred'] = result.predict(combos[train_cols])
    combos.to_csv("Results.csv", sep=',', encoding='utf-8')

# Calculating the std of users' probability o being in the treated group.
# This number will be used to calculate the penalties while lookinf for proper matched users
def GetSD():
    df = pd.read_csv("Results.csv")
    SD = df['admit_pred'].std()
    print(df.describe())
    return SD
# Creating distance matrices. This matrice shows the distance
# between every two user, one in the treated group and other in the control group
def MakePSdistanceMatrice():
    df = pd.read_csv("Results.csv")
    counter= ((df['Admited'].value_counts()))
    treatedNum = counter[1]
    nontreatedNum = counter[0]
    allParticipants = treatedNum+nontreatedNum
    PSMatrice = np.zeros((treatedNum, allParticipants))
    treatedUsers=[]
    for x in range(df['UserId'].count()):

        if str(df['Admited'][x]) == '1':
            treatedUsers.append(df['UserId'][x])
    for x in range(len(treatedUsers)):
        inde = df['UserId'][df['UserId']==treatedUsers[x]].index[0]
        ex1 = df['admit_pred'][inde]
        for y in range(df['UserId'].count()):
            # A user cannot matche with itself
            if str(df['UserId'][inde]) == str(df['UserId'][y]):
                PSMatrice[x,y]=-1
            else:
                ex2 = df['admit_pred'][y]
                PSMatrice[x, y]= abs(ex1 - ex2)

    f = open('PropensityScores.csv', 'w')
    f.write('UserIds')
    for x in range(df['UserId'].count()):
        f.write(',')
        f.write(str(df['UserId'][x]))
    f.write('\n')
    for x in range(len(treatedUsers)):
        f.write(str(treatedUsers[x]))
        for y in range(allParticipants):
            f.write(',')
            f.write(str(PSMatrice[x,y]))
        f.write('\n')
    f.close()
    print(PSMatrice)
# This function add penalty to the distances between couple of users,
#  one in the treated group and second in the control group, if their
#  distance is larger than the half of std of users' probability of being in the treated group
def CalculatePenalties():
    df = pd.read_csv("PropensityScores.csv")
    f = open('PropensityScoresPenalties.csv', 'w')
    standardD = GetSD()
    w = standardD/2.0
    f.write(str(df.columns.values[0]))
    for x in range(1,len(df.columns.values)):
        f.write(',')
        f.write(str(df.columns.values[x]))
    f.write('\n')
    for x in range(df['UserIds'].count()):
        f.write(str(df.iat[x,0]))
        for y in range(1,len(df.columns)):
            f.write(',')
            if df.iat[x,y] > w:
                penalty = 1000*(df.iat[x,y]-w)
                f.write(str(penalty))
            else:
                f.write('0')
        f.write('\n')
    f.close()

# Finding the best matched user from control group for every user in the treated group
def findMatchedUser():
    df = pd.read_csv("PropensityScoresPenalties.csv",index_col=0)
    dftreat = pd.read_csv("treatedUser,SII.csv", index_col=0)
    print(dftreat.index[2])
    dfcontrol = pd.read_csv("allControl.csv", index_col=0)
    f = open('MatchedUsers.txt', 'w')
    counter = 0
    penalty=0
    for x in range(dftreat['Year'].count()):
        treatVector = [dftreat.iat[x, 2],dftreat.iat[x, 3]]
        bestMatch = 1000
        bestId = 0
        print(counter)
        counter+=1
        for y in range(dfcontrol['Date'].count()):
            print(y)
            controlVector = [dfcontrol.iat[y,1]], dfcontrol.iat[y,2]
            penalty = float(df[str(dfcontrol.index[y])].loc[[dftreat.index[x]]])
            dis = penalty
            if dis<bestMatch:
                bestMatch=(dis)
                bestId=y
        f.write(str(dftreat.index[x]))
        f.write((','))
        f.write(str(dfcontrol.index[bestId]))
        f.write('\n')
    f.close()

        
            

if __name__ == "__main__":
    Treated="Treated_Users.txt"
    makeMatchInput(Treated)
    GetPS()
    MakePSdistanceMatrice()
    CalculatePenalties()
    findMatchedUser()

