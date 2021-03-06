import pandas as pd
import numpy as np
from keras.utils import get_file
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score
import matplotlib.pyplot as plt
import itertools

traffic = get_file('kddcup.data_10_percent.gz', origin='http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz')
data = pd.read_csv(traffic, header = None)
data.dropna(inplace=True, axis=1)
data.columns = ['Time', 'Prot_Type', 'Service', 'Flag', 'Src_bytes', 'Dst_bytes', 'Land', 'Wrong_Fragment', 'Urgent', 'Hot',
    'Failed_Log_ins', 'Log_Ins', 'Compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'Count', 'Srv_Count', 'Serror_Rate', 'Srv_Serror_Rate',
    'Rerror_Rate', 'Srv_Rerror_Rate', 'Same_Srv_Rate', 'Diff_Srv_Rate', 'Srv_Diff_Host_Rate', 'Dst_Host_Count',
    'Dst_Host_Srv_Count', 'Dst_Host_Same_Srv_Rate', 'Dst_Host_Diff_Srv_Rate',
    'Dst_Host_Same_Src_Port_Rate', 'Dst_Host_Srv_Diff_Host_Rate', 'Dst_Host_Serror_Rate', 'Dst_Host_Srv_Serror_Rate',
    'Dst_Host_Rerror_Rate', 'Dst_Host_Srv_Rerror_Rate', 'Result']

# Attack Vs. Normal

data.replace(
to_replace=['ipsweep.', 'portsweep.', 'nmap.', 'satan.', 'ftp_write.', 'guess_passwd.', 'imap.', 'multihop.',
'phf.', 'spy.', 'warezclient.', 'warezmaster.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.', 'back.',
 'land.', 'neptune.', 'pod.', 'smurf.', 'teardrop.'], value='Attack', inplace=True)

data.replace(to_replace=['normal.'], value='Normal', inplace=True)

data.groupby('Result')['Result'].count()

#  Normal, R2L, U2R, Probe, DOS

data.replace(to_replace=['normal.'], value = 'Normal', inplace = True)
data.replace(to_replace = ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'phf.', 'spy.', 'warezclient.', 'warezmaster.'], value = 'R2LAttack', inplace = True)
data.replace(to_replace = ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.'], value = 'U2RAttack', inplace = True)
data.replace(to_replace = ['ipsweep.', 'portsweep.', 'nmap.', 'satan.'], value = 'ProbeAttack', inplace = True)
data.replace(to_replace = ['back.', 'land.' , 'neptune.', 'pod.', 'smurf.', 'teardrop.'], value = 'DOSAttack', inplace = True)
data.groupby('Result')['Result'].count()

le = LabelEncoder()
data['Prot_Type'] = le.fit_transform(data['Prot_Type'])
data['Flag'] = le.fit_transform(data['Flag'])
data['Service'] = le.fit_transform(data['Service'])

X = data.copy().drop(['Result'], axis=1)
y = data['Result']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

DT = DecisionTreeClassifier(criterion='gini', class_weight='balanced')
pred = DT.fit(X_train, Y_train).predict(X_test)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Decision Tree',
                          cmap=plt.cm.Greys):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

CM = confusion_matrix(Y_test, pred)
plot_confusion_matrix(CM,['DOS Attack', 'Normal', 'Probe Attack', 'R2L Attack', 'U2R Attack'])
plt.show()

AS = accuracy_score(Y_test, pred)
print('Accuracy Score: ')
print(AS)
print('--' * 50)
PS = precision_score(Y_test, pred, average='micro')
print('Precision Score:')
print(PS)
print('--' * 50)
f1 = f1_score(Y_test, pred, average='micro')
print('F1 score:')
print(f1)
print('--' * 50)
CR = classification_report(Y_test, pred)
print('Classification Report:')
print(CR)
print('--' * 50)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (Y_test != pred).sum()))
print(pred[1])
print(Y_test.iloc[1])
