import pandas as pd
import numpy as np
from keras.utils import get_file
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import statistics

traffic = get_file('kddcup.data_10_percent.gz',
                       origin='http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz')

data = pd.read_csv(traffic, header = None)

# Adds header to data
data.columns = ['Time', 'Prot_Type', 'Service', 'Flag', 'Src_bytes', 'Dst_bytes', 'Land', 'Wrong_Fragment', 'Urgent', 'Hot',
    'Failed_Log_ins', 'Log_Ins', 'Compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'Count', 'Srv_Count', 'Serror_Rate', 'Srv_Serror_Rate',
    'Rerror_Rate', 'Srv_Rerror_Rate', 'Same_Srv_Rate', 'Diff_Srv_Rate', 'Srv_Diff_Host_Rate', 'Dst_Host_Count',
    'Dst_Host_Srv_Count', 'Dst_Host_Same_Srv_Rate', 'Dst_Host_Diff_Srv_Rate',
    'Dst_Host_Same_Src_Port_Rate', 'Dst_Host_Srv_Diff_Host_Rate', 'Dst_Host_Serror_Rate', 'Dst_Host_Srv_Serror_Rate',
    'Dst_Host_Rerror_Rate', 'Dst_Host_Srv_Rerror_Rate', 'Result']

# Categorization 1
#data.replace(
#to_replace=['ipsweep.', 'portsweep.', 'nmap.', 'satan.', 'ftp_write.', 'guess_passwd.', 'imap.', 'multihop.',
#'phf.', 'spy.', 'warezclient.', 'warezmaster.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.', 'back.',
#'land.', 'neptune.', 'pod.', 'smurf.', 'teardrop.'], value='Attack', inplace=True)

#data.replace(to_replace=['normal.'], value='Normal', inplace=True)
#data.groupby('Result')['Result'].count()

# Categorization 2

data.replace(to_replace=['normal.'], value = 'Normal', inplace = True)
data.replace(to_replace = ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'phf.', 'spy.', 'warezclient.', 'warezmaster.'], value = 'R2LAttack', inplace = True)
data.replace(to_replace = ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.'], value = 'U2RAttack', inplace = True)
data.replace(to_replace = ['ipsweep.', 'portsweep.', 'nmap.', 'satan.'], value = 'ProbeAttack', inplace = True)
data.replace(to_replace = ['back.', 'land.' , 'neptune.', 'pod.', 'smurf.', 'teardrop.'], value = 'DOSAttack', inplace = True)
print(data.groupby('Result')['Result'].count())

plt.figure(figsize=(12, 10))
par = {'axes.titlesize': '15',
       'ytick.labelsize': '12',
       'xtick.labelsize': '12'}
plt.rcParams.update(par)
plt.title('Traffic distribution')
data['Result'].value_counts().apply(np.log).plot(kind='bar')
