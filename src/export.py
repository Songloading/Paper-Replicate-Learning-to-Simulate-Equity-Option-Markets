import sys
import os
from google.colab import files

for i in range(len(dlv)):
  filename = 'spxw_call_dlv_' + str(i + 26) +'.csv'
  with open('/data/dlv/' + filename, 'w') as f:
    np.savetxt(f, dlv[i], delimiter=",")
