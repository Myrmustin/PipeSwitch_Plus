"""

The data for Table 1 is in the arrays ps1, ps2, ps3
Table 1 : 
for every Inception task ( Resnet, BertBase,Inception) we show:

for example :
Inception:
speed of inference request
speed of inference if previously we trained on the same model
speed of infer. if we trained on bert_base 
speed of infer if previously on Resnet 

Bert_base:
(same)
(same)
if we trained Inception
if we trained ResNet152


ResNet152
(same)
(same)
if we trained Inception
if we trained bert_base




the data for this is in array: ps_Ready_RequestA_Normal
Table 2: 
We test regular PS VS request aware PipeSwitch 

Bert_base:
	ready model: 
	RequestWarePS
	PS: 
Inception 
      Same
      Same
      Same
Resnet. 
      Same
      Same
      Same

	

"""




import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

ps = [['ResNet18', 46.594021, 41.872554, 7260.390292],
      ['ResNet34', 71.991523, 69.879479, 7614.991212],
      ['ResNet50', 84.444547, 82.951148, 7727.970052],
      ['ResNet101', 143.609150, 141.310623, 8447.313666],
      ['ResNet152', 201.973748, 198.974363, 9024.306250],
      ['ResNeXt50_32x4d', 247.893031, 237.266811, 7742.501783],
      ['ResNeXt101_32x8d', 678.810241, 645.620963, 9655.950654]]

ps1 = ['Resnet152', 76.353168, 80.487001, 85.94260215759277, 96.08784198760986 ]
ps2 = ['Bert_base', 55.139515, 62.01514, 59.23572540283203, 61.74659729003906]
ps3 = ['Inception_v3', 65.15407, 65.726674, 84.42037105560303 , 82.52977848052979]     


ps_Ready_RequestA_Normal = [
      ['ResNet152' , 68.31, 67.94820672426468, 76.38671557108562],
      ['Bert_Base' , 45.70 , 47.68399917162382, 56.91155195236206],
      ['Inception', 57.112047 , 54.90562457304734, 63.844192822774254]]

ps_Complex1_Request_PipeSwitch = []

      
for x in ps:
      print(x[0],x[1] - x[2])

df = pd.DataFrame(ps1, columns=['Inference', 'Switch from ResNet152', 'Switch from Inception', 'Switch from Bert_base'])

df.plot(x='Model', y=['Inference', 'Switch from ResNet152', 'Switch from Inception', 'Switch from Bert_base'], kind='bar',figsize=(7,4),  rot=45)
#plt.yscale('log')
plt.ylabel('Latency [ms]')
plt.legend(bbox_to_anchor=(1,1), loc='best', ncol=3)
plt.ylim(1,10.1e4)

plt.show()
# plt.savefig('fig.png', bbox_inches="tight", dpi=300)