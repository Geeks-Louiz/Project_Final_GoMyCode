#************************************************ Bibliotheque*****************************************************************
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#************************************************* Importing_Data ************************************************************
data = pd.read_csv('C:\\Users\\belar\\Desktop\\untitled\\heart.csv')
print(data.head(5))    # to show only five rows from the dataset.
print(data.nunique(axis=0))
print(data.describe())
# ************************************************ Checking for missing values ************************************************
print(data.isnull().sum())
print(data.isnull().values.any())

# ************************************************* Data Exploration ************************************************************
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,fmt='.1f')
plt.show()

plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)
plt.tight_layout()
plt.show()

#************************************************ Analysis***********************************************************************
# Age analysis
minAge=min(data.age)
maxAge=max(data.age)
meanAge=data.age.mean()
print('Age_Minimum :',minAge)
print('Age_Maximum :',maxAge)
print('Moyenne_Age :',meanAge)
# Range age
young_ages=data[(data.age>=29)&(data.age<40)]
middle_ages=data[(data.age>=40)&(data.age<55)]
elderly_ages=data[(data.age>55)]
print('Interval_Age_jeune :',len(young_ages))
print('Interval_Age_moyen :',len(middle_ages))
print('Interval_Age_avancé :',len(elderly_ages))

sns.barplot(x=['Interval_Age_jeune','Interval_Age_moyen','Interval_Age_avancé'],y=[len(young_ages),len(middle_ages),len(elderly_ages)])
plt.xlabel('Interval_Age')
plt.ylabel('Total_Personne')
plt.title('Interval_Age_Dataset')
plt.show()

data['AgeRange']=0
youngAge_index=data[(data.age>=29)&(data.age<40)].index
middleAge_index=data[(data.age>=40)&(data.age<55)].index
elderlyAge_index=data[(data.age>55)].index
for index in elderlyAge_index:
    data.loc[index, 'AgeRange'] = 2

for index in middleAge_index:
    data.loc[index, 'AgeRange'] = 1

for index in youngAge_index:
    data.loc[index, 'AgeRange'] = 0

sns.swarmplot(x="AgeRange", y="age", hue='sex',
              palette=["r", "c", "y"], data=data)
plt.show()

# how many mal or femal in each range
sns.countplot(elderly_ages.sex)
plt.title("Distribution de la tranche age_avancée en fonction du sexe de la personne")
plt.show()

Sex_Thalach=(elderly_ages.groupby(elderly_ages['sex'])['thalach'].agg('sum'))
sns.barplot(x=elderly_ages.groupby(elderly_ages['sex'])['thalach'].agg('sum').index,y=elderly_ages.groupby(elderly_ages['sex'])['thalach'].agg('sum').values)
plt.title("Frequence_cardiaque_maximal par groupe de sexe")
plt.show()

colors = ['blue','green','yellow']
explode = [0,0,0.1]
plt.figure(figsize = (5,5))
#plt.pie([target_0_agerang_0,target_1_agerang_0], explode=explode, labels=['Target 0 Age Range 0','Target 1 Age Range 0'], colors=colors, autopct='%1.1f%%')
plt.pie([len(young_ages),len(middle_ages),len(elderly_ages)],labels=['young ages','middle ages','elderly ages'],explode=explode,colors=colors, autopct='%1.1f%%')
plt.title('Interval_Age',color = 'blue',fontsize = 15)
plt.show()

# Sex Analysis
# 1: male 0:femme   on a plus d'homme 207 que de femmes 96
print(data.sex.value_counts())
sns.countplot(data.sex)
plt.show()
# SLOP La pente de la relation segment ST / fréquence cardiaque pendant l'exercice  0: descendant; 1: plat; 2: montée
sns.countplot(data.sex,hue=data.slope)
plt.title('la distribution de slope sachant le sexe de la population')
plt.show()
#Selon cet ensemble de données de Cleveland, les hommes sont plus susceptibles de contracter une maladie cardiaque que les femmes. Les hommes subissent plus de crises cardiaques que les femmes. Les crises cardiaques soudaines sont vécues par les hommes entre 70% et 89%. La femme peut avoir une crise cardiaque sans aucune pression thoracique, elle éprouve généralement des nausées ou des vomissements qui sont souvent confondus avec le reflux acide ou la grippe.
total_genders_count=len(data.sex)
male_count=len(data[data['sex']==1])
female_count=len(data[data['sex']==0])
#pourcentage male/female
print("Taux sexe Homme: {:.2f}%".format((male_count / (total_genders_count)*100)))
print("Taux de sexe Femme: {:.2f}%".format((female_count / (total_genders_count)*100)))

#Male State & target 1 & 0
male_andtarget_on=len(data[(data.sex==1)&(data['target']==1)])
male_andtarget_off=len(data[(data.sex==1)&(data['target']==0)])
####
sns.barplot(x=['Homme atteind de maladie cardiaque','Homme sain'],y=[male_andtarget_on,male_andtarget_off])
plt.xlabel('Distribution dhommes atteints de malade cardiaque')
plt.ylabel('Nombre')
plt.title('Sexe')
plt.show()
#Female State & target 1 & 0
female_andtarget_on=len(data[(data.sex==0)&(data['target']==1)])
female_andtarget_off=len(data[(data.sex==0)&(data['target']==0)])
####
sns.barplot(x=['Femmes atteintes de maladie cardiaque','Femmes saines'],y=[female_andtarget_on,female_andtarget_off])
plt.xlabel('Distribution du nombre de femmes atteintes de maladie cardiaque')
plt.ylabel('Nombre')
plt.title('Sexe')
plt.show()

# Chest Pain Analysis
print(data.cp.value_counts())
sns.countplot(data.cp)
plt.xlabel('Chest Pain')
plt.ylabel('Nombre')
plt.title('Nombre de chaque type de Chest Pain')
plt.show()

cp_zero_target_zero=len(data[(data.cp==0)&(data.target==0)])
cp_zero_target_one=len(data[(data.cp==0)&(data.target==1)])
sns.barplot(x=['Nombre de patient asymptomatique et non atteints','Nombre de patients asymptique mais atteints de maladie cardiaque'],y=[cp_zero_target_zero,cp_zero_target_one])
plt.show()

cp_one_target_zero=len(data[(data.cp==1)&(data.target==0)])
cp_one_target_one=len(data[(data.cp==1)&(data.target==1)])
sns.barplot(x=['Nombre de patient agine_atypique et non atteints','Nombre de patients agine atypique mais atteints de maladie cardiaque'],y=[cp_one_target_zero,cp_one_target_one])
plt.show()

cp_two_target_zero=len(data[(data.cp==2)&(data.target==0)])
cp_two_target_one=len(data[(data.cp==2)&(data.target==1)])
sns.barplot(x=['Nombre de patient pas dangine et non atteints','Nombre de patients pas dangine mais atteints de maladie cardiaque'],y=[cp_two_target_zero,cp_two_target_one])
plt.show()

cp_three_target_zero=len(data[(data.cp==3)&(data.target==0)])
cp_three_target_one=len(data[(data.cp==3)&(data.target==1)])
sns.barplot(x=['Nombre de patient avec angine severe et non atteints','Nombre de patients avec angine severe mais atteints de maladie cardiaque'],y=[cp_three_target_zero,cp_three_target_one])
plt.show()

data.thalach.value_counts()[:20]
sns.barplot(x=data.thalach.value_counts()[:20].index,y=data.thalach.value_counts()[:20].values)
plt.xlabel('thalach')
plt.ylabel('Nombre')
plt.title('Distribution de chaque frequence cardiaque')
plt.xticks(rotation=45)
plt.show()

# THAL analysis
data.thal.value_counts()
sns.countplot(data.thal)
plt.show()

a=len(data[(data['target']==1)&(data['thal']==0)])
b=len(data[(data['target']==1)&(data['thal']==1)])
c=len(data[(data['target']==1)&(data['thal']==2)])
d=len(data[(data['target']==1)&(data['thal']==3)])
print('target 1 thal 0: ',a)
print('target 1 thal 1: ',b)
print('target 1 thal 2: ',c)
print('target 1 thal 3: ',d)

#so,Apparently, there is a rate at Thal 2.Now, draw graph
print('*'*50)
#Target 0
e=len(data[(data['target']==0)&(data['thal']==0)])
f=len(data[(data['target']==0)&(data['thal']==1)])
g=len(data[(data['target']==0)&(data['thal']==2)])
h=len(data[(data['target']==0)&(data['thal']==3)])
print('target 0 thal 0: ',e)
print('target 0 thal 1: ',f)
print('target 0 thal 2: ',g)
print('target 0 thal 3: ',h)

f,ax=plt.subplots(figsize=(7,7))
sns.barplot(y=['T 1&0 Th 0','T 1&0 Th 1','T 1&0 Th 2','Ta 1&0 Th 3'],x=[1,6,130,28],color='green',alpha=0.5,label='target 1 Thal State')
sns.barplot(y=['T 1&0 Th 0','T 1&0 Th 1','T 1&0 Th 2','Ta 1&0 Th 3'],x=[1,12,36,89],color='red',alpha=0.7,label='target0 Thal State')
ax.legend(loc='lower right',frameon=True)
ax.set(xlabel='Atteinte cardiaque et Taux trouble sanguin',ylabel='Etat datteinte cardiaque sachant le trouble sanguin',title='target VS Thal')
plt.xticks(rotation=90)
plt.show()

#Target Analysis
data.target.unique()
sns.countplot(data.target)
plt.xlabel('target')
plt.ylabel('Nombre')
plt.title('Distribution de attribut Target 1 & 0')
plt.show()

sns.countplot(data.target,hue=data.sex)
plt.xlabel('target')
plt.ylabel('Nombre')
plt.title('Distribution de Traget sachant le sexe ')
plt.show()

