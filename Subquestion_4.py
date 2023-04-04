from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report
import Subquestion_1
import pandas as pd

df = Subquestion_1.result_dict
df_4 = pd.DataFrame.from_dict(df, orient='index')

