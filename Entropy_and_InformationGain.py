
import pandas as pd
import numpy as np

edu_data_read = pd.read_excel("edu_data.xlsx")

## 1-4-7-10... sekilde hangi satirla ilgilendigim
edu_data = edu_data_read.iloc[1::3] 
edu_data_copy = edu_data.copy()

## indexleri 1 den 160'a gidecek sekilde guncelledik
edu_data.index = pd.RangeIndex(start = 0,stop = 160, step = 1)

"""
'gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'SectionID',
'Topic', 'Semester', 'Relation', 'raisedhands', 'AnnouncementsView',
'Discussion', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction',
'StudentAbsenceDays', 'Class'
"""
column = edu_data.shape[0]
row = edu_data.shape[1]

pd.options.mode.chained_assignment = None

###################################### Madde 1 #################################################################
def verileri_at(dataframe, column_name):
    """
    Summary :
         Verinin merkezinden iki standart sapma uzaklıktaki verileri nan olarak ata.
         Daha sonra nan olarak atanan verileri mod ile doldur.
    
    Args:
        column_name (str): 
            dataframe'in kolonunun adı
        dataframe (DataFrame):
            verinin merkezinden iki standart sapma uzağındaki atılacak verilerin dataframe'i
            
    Returns:
        data.frame --> edu_data
    """
    
    # 𝜇 kolonun ortalaması 
    kolonun_ortalaması = int(dataframe[column_name].mean())
    # 𝜎 kolonun standart sapması
    kolonun_std = int(dataframe[column_name].std()) 
    
    # merkezden iki standart sapma ilerisinde ki verinin sayısal değeri
    mu_arti_ikisigma = (int(kolonun_ortalaması) + 2 * (int(kolonun_std)))
    
    # merkezden iki standart sapma gerisinde ki verinin sayısal değeri
    mu_eksi_ikisigma = (int(kolonun_ortalaması) - 2 * (int(kolonun_std)))
    
    # merkezden iki standart sapma ileriside ki verinin sayısal değerden büyük olanlara nan ata
    dataframe.loc[dataframe[column_name] > mu_arti_ikisigma, column_name] = np.nan
    
    # merkezden iki standart sapma gerisinde ki verinin sayısal değerden kücük olanlara nan ata
    dataframe.loc[dataframe[column_name] < mu_eksi_ikisigma, column_name] = np.nan
    
    # ilgili kolonun en cok tekrar eden degerini bul
    mode = dataframe.loc[:, [column_name]].mode()
    
    # na veri iceren kolona yukarıda ki degiskende bulunan mod degerini ata
    dataframe.loc[dataframe[column_name].isna(), [column_name]] = mode.values
    
    return dataframe


# veri setinden sadece numeric veriler iceren kolonları al
numeric_veriler = [x for x in edu_data.columns if (edu_data[x].dtypes == "int64") or 
                                                 (edu_data[x].dtypes == "float64")]

# numeric veriler iceren kolonların hepsi icin metodu(verileri_at) uygula
for i in edu_data[numeric_veriler].columns:
    verileri_at(edu_data, edu_data[i].name)


arrayy = []
for i in edu_data[numeric_veriler].columns:
    arrayy.append(edu_data.loc[:,  i].mode().values)

mode = pd.DataFrame(arrayy)

veri_ozeti = pd.DataFrame([edu_data.describe().T.loc[:, "mean"], edu_data.describe().T.loc[:, "std"]]).T
veri_ozeti.insert(2, "mod", mode)
################################################################################################################

###################################### Madde 2 #################################################################

sum_no_column_2 = ["gender", "Semester", "Relation", "ParentAnsweringSurvey", "ParentschoolSatisfaction", "StudentAbsenceDays"]
sum_no_column_3 = ["StageID", "SectionID", "Class"]
sum_no_column_12 = ["Topic", "PlaceofBirth", "NationalITy"]


import matplotlib.pyplot as plt

for i in edu_data[sum_no_column_2].columns:
    bins = np.arange(10) - 0.5
    plt.figure(figsize = (12,5))
    plt.hist(x = edu_data[i], bins = bins, edgecolor = "black")    
    plt.xticks(range(10), fontsize = 13)
    plt.xlabel(edu_data[i].name, weight='bold', fontsize = 13)
    plt.xlim([-1, 2])
    plt.ylim(0, 110)
    plt.show()



for j in edu_data[sum_no_column_3].columns:
    bins = np.arange(10) - 0.5
    plt.figure(figsize = (12,5))
    plt.hist(x = edu_data[j], bins = bins, edgecolor = "black")
    plt.xticks(range(10), fontsize = 13)
    plt.xlabel(edu_data[j].name, weight='bold', fontsize = 12)
    plt.xlim([-1, 3])
    plt.ylim(0, 120)
    
    
for k in edu_data[sum_no_column_12].columns:
    bins = np.arange(10) - 0.5
    plt.figure(figsize = (12,5))
    plt.hist(x = edu_data[k], bins = bins, edgecolor = "black")
    plt.xticks(range(10), fontsize = 11)
    plt.xlabel(edu_data[k].name, weight='bold', fontsize = 12)
    plt.xlim([-1, 10])
    plt.ylim(0, 120)
    
    
    
kategorik_veriler = [x for x in edu_data.columns if edu_data[x].dtypes == "O"]
# mode_ile_degistir metodu kullanmadan önce 
for i in edu_data[kategorik_veriler].columns:
     print(edu_data[i].value_counts(), "\n\n")
     
     
def mode_ile_degistir(dataframe, column_name, threshold = 3):
    """
    Summary : 
    
    Histogramı cizdirilen verisetinde az gorulen degerleri
    en cok tekrar eden degerle degistir(mod).

    Args:
        column_name (str) : dataframe'in kolonunun adı
        threshold (int64) : en az kac deger mod ile degistirilsin, default = 2
        
    Returns:
        data.frame --> edu_data
    """
    
    # threshold'dan az gorulen degerler
    c_data = dataframe[column_name].value_counts() <= threshold
    
    # threshold'dan az gorulen degerlerin adları
    silinecek_verilerin_adları = c_data.index[c_data == True]
    
    # mod bulma(daha sonra en cok tekrar edenle degistirmek icin)
    mode = dataframe.loc[:, [column_name]].mode() 
    
    # silinecek verilerin adlari kadar donen bir for
    for i in range(len(silinecek_verilerin_adları)): 
        
        #degistirelecek verileri tespit etme
        degistirelecek = dataframe[dataframe[column_name] == silinecek_verilerin_adları[i]]
        
        #degistirelecek verileri kolonda en cok tekrar eden degerle(mod) ile degistirme
        dataframe.loc[degistirelecek.index, [column_name]] = mode.values
        
    return dataframe

# mode_ile_degistir metodunu tüm kolonlar icin uygula
for i in edu_data[kategorik_veriler].columns:
    mode_ile_degistir(edu_data, edu_data[i].name)


     
# mode_ile_degistir metodu kullandıktan sonra
for i in edu_data[kategorik_veriler].columns:
     print(edu_data[i].value_counts(), "\n\n")


    
################################################################################################################
#################################### Madde 3 ###################################################################
import math
from scipy.stats import entropy

kategorik_veriler = [x for x in edu_data_copy.columns if edu_data_copy[x].dtypes == "O"]


entropy_list = []

def entropy_calculate(dataframe, column_name):
    """
    Summary:
        Veri setinin tüm kolonlarının entropsini bulup
        sırasıyla liste icine ekliyor

    Args:
        dataframe (data.Frame): Veri seti
        column_name (str): kolonun adı
    
    Raises :
        Exception :
            Eger object veri tipi dışında bir
            veri tipi girilirse exception fırlatır.
        
    Returns:
        list : kolonların entropileri      
    """
    
    # Kolon tipi Object ise islem yap
    if dataframe[column_name].dtypes == "O":
        
        # ilgili kolonun unique degerlerinin toplamını bul
        c_data_copy = dataframe[column_name].value_counts()
        
        # entropy diye bir degiskende entropy degerlerini toplayacağız
        entropy = 0
        
        len_of_cdata = len(c_data_copy)
        
        # unique degerlerin length'i kadar donen bir for dongusu
        for i in range(len_of_cdata):
             
            # ilgili kolonun her bir unique degerini df'in satırına bolerek olasılıgını bulduk
            p_of_val = (c_data_copy[i] / dataframe.shape[0])
            
            #entropiyi hesapladık
            entropy += -(math.log(p_of_val, 2) * p_of_val)
            
            #eger bir kolondaki unique degerlerin hepsi icin entropy hesapladiysa diziye ekle
            if i == len_of_cdata - 1: # eger hesaplama bitmediyse for da donmeye devam et
                entropy_list.append(entropy)
    else:
        raise Exception("Sadece kategorik verilerin entropisi hesaplanır.")
        
    return entropy_list


for i in edu_data_copy[kategorik_veriler].columns:
    entropy_calculate(edu_data_copy, edu_data_copy[i].name)

# Kolon entropilerini dataframe'e cevirme
ozelliklerin_Entropileri = pd.DataFrame(index = [kategorik_veriler], data = entropy_list).sort_values(by = [0])
ozelliklerin_Entropileri.rename(columns={0:"Entropiler"})


################################################################################################################
#################################### Madde 4  ##################################################################


Information_Gain_List = []
entropy_list = []
class_haric = kategorik_veriler[:-1]

def calculate_column_IG(dataframe, column_name, class_label):
    """
    Summary:
        DataFrame'i kolonu ve class label'i verilen
        sütunların information gainini hesaplar
    
    Args:
        dataframe (dataFrame): dataFrame
        column_name (object): kolonun ismi
        class_label (object): hangi class label kullanılacağı
    
    Raises:
        Exception:
        Eger kategorik olmayan bir kolon girilirse
        exception doner
        
    Returns:
        list: 
        Sırasıyla sütunların information gainini doner
"""
    
    # Kolonun entropisini tutmak icin bir degisken
    IG = 0    
    
    # Eger kategorik olmayan bir kolon girilirse exception doner
    if dataframe[column_name].dtypes != 'O':
        raise Exception("Kategorik olmayan kolon goruldu.")
    else:
        
        # kolonlardaki toplam unique degerleri
        kolonun_nunique_sayisi = dataframe[column_name].nunique()
        
        # kolonun_nunique_sayisi kadar donen bir for
        for j in range(kolonun_nunique_sayisi): 
            
            # hic na deger var mi ?
            if dataframe.isna().values.any(): 
                dataframe = dataframe.dropna() # varsa dusur
            else:
                # ilgili kolonun her bir unique degerinin class label ile ikili kombinasyonunu al bir data frame'e cevir 
                hesaplanmak_istenen_sutun = pd.DataFrame(dataframe.loc[dataframe[column_name] == dataframe.loc[:,column_name].unique()[j]][[column_name, class_label]])
                
                """
                gender  Class
                M       M        50
                        L        37
                        H        19
                """
                # vc_of_his icin yukarıda gender kolonunun 'M' degeri icin bir cikti var
                # gender 'M' iken Class icin ['M', 'L', 'H'] degerlerini getirtir
                vc_of_his = pd.DataFrame(hesaplanmak_istenen_sutun.value_counts())
                
                # Hesaplanmak istenen sutunların entropilerini hesaplattir
                Entropy_of_his = entropy_calculate(hesaplanmak_istenen_sutun, hesaplanmak_istenen_sutun[class_label].name)
                
                # bir sutünun her bir unique degerini level diye bir degiskende tut
                level = dataframe.loc[:,column_name].unique()[j]
                
                # yukarıda verilen ornekde ki gibi tum degerlerin teker teker toplamini bul (gender ornegi)
                SUM = vc_of_his.xs(level)[0].sum() # multindex oldugu icin xs ile eristik
                row_of_df = dataframe.shape[0]
                
                # her bir kolonun unique degerlerinin olasiligini bul ## gender icin ornek (50+37+19) / row_of_df
                prob = SUM / row_of_df
                
                # Entropy_of_his'e son eklenen degerle olasılıgını carp ve ilgili kolonun entropy'sini bul
                IG += (Entropy_of_his[-1] * prob)
                #print(level," | ", SUM, " | ", Entropy_of_his[-1], " * (", SUM, "/",row_of_df, ") --> ", IG, "\n")
                
                # eger kolon icin son unique degere geldiyse hesaplama bitti listeye ekle
                if j == kolonun_nunique_sayisi - 1:
                    Information_Gain_List.append(IG)
                    IG = 0 # sonraki kolon icin yeniden IG hesaplamak icin sifirladik
                    
    return print(Information_Gain_List[-1])

for i in edu_data[class_haric].columns:
    calculate_column_IG(edu_data, edu_data[i].name, "Class")
    
print(Information_Gain_List)    

################################################################################################################
#################################### Madde 5 ###################################################################

# Yeni numeric_verilerin bulunduğu veri seti
numeric_veriler = [x for x in edu_data.columns if (edu_data[x].dtypes == "int64") or 
                                                (edu_data[x].dtypes == "float64")]

new_df_for_PCA = edu_data_copy[numeric_veriler]
new_df_for_PCA


# Normalization part
from sklearn.preprocessing import StandardScaler

new_df_for_PCA = StandardScaler().fit_transform(new_df_for_PCA.values)
new_df_for_PCA


## PCA uygulanan kısım
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)

principalComponents = pca.fit_transform(new_df_for_PCA)

after_PCA_edu_data = pd.DataFrame(data = principalComponents, columns = ["PCA1", "PCA2"])
after_PCA_edu_data


# PCA sonrası scatter plot cizimi
import seaborn as sns

sns_plot = sns.scatterplot(data = after_PCA_edu_data)



################################################################################################################

#################################### Madde 6  ##################################################################

# Knn icin PCA uygulanan veri setinden X ve Y'leri cekme
X = after_PCA_edu_data.iloc[:, [0, 1]]
y = edu_data.loc[:, "Class"]


#Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 42)
print(len(X_test),len(X_train))

# StandartScaler uygulanan kısım
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Training seti KNN üzerinde egitme
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)


# Test Seti predict etme
y_pred = classifier.predict(X_test)


# Accuracy
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)

print(f"Accuracy with: {ac}")
