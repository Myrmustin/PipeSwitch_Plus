import sys
import time
import struct
import statistics

from task.helper import get_data
from util.util import TcpClient, timestamp
import pickle
import os


def main():

    #using Exodia2
    list_bertBase = [600, 600, 
    #Values from 5x10 runs x6 requests each
    272.8419303894043, 273.3771800994873, 272.8574275970459, 275.7105827331543, 272.45187759399414, 273.18811416625977, 272.1295356750488, 272.8891372680664, 272.676944732666, 272.7999687194824
    , 273.55360984802246, 274.17874336242676, 274.14941787719727, 273.58055114746094, 272.16410636901855, 273.303747177124, 273.2970714569092, 273.4029293060303, 272.6747989654541, 273.2524871826172
    , 273.3118534088135, 272.82094955444336, 273.76270294189453, 272.6266384124756, 274.1267681121826, 272.02773094177246, 273.1032371520996, 272.66454696655273, 272.91154861450195, 272.5861072540283
    , 274.0349769592285, 273.5445499420166, 273.83899688720703, 273.90027046203613, 273.8528251647949, 275.76255798339844, 278.40280532836914, 273.46086502075195, 277.3454189300537, 272.9358673095703
    ,274.4336128234863, 273.44393730163574, 272.7956771850586, 275.2096652984619, 273.11086654663086, 273.78058433532715, 273.6356258392334, 273.1940746307373, 273.1800079345703, 273.12231063842773
    ]

    list_Resnet = [600,600
    , 408.282995223999, 398.9229202270508, 397.6635932922363, 393.09167861938477, 387.3894214630127, 397.1283435821533, 405.34257888793945, 395.58982849121094, 392.7454948425293, 392.017126083374
    , 411.90361976623535, 403.1076431274414, 407.98187255859375, 397.8121280670166, 396.82888984680176, 397.77350425720215, 404.9689769744873, 403.3622741699219, 398.6055850982666, 400.7852077484131
    , 398.47874641418457, 403.4247398376465, 420.23706436157227, 401.59106254577637, 418.75243186950684, 392.9553031921387, 408.0352783203125, 401.03602409362793, 406.16536140441895, 397.79210090637207
    , 424.0298271179199, 394.8845863342285, 407.14406967163086, 391.62230491638184, 394.03414726257324, 399.43552017211914, 400.30956268310547, 390.028715133667, 405.80201148986816, 391.6172981262207
    , 407.55224227905273, 390.2139663696289, 398.8533020019531, 392.79842376708984, 398.146390914917, 390.6872272491455, 387.94851303100586, 397.8309631347656, 399.3351459503174, 397.7944850921631
    ]
    list_inception = [ 600, 600
    , 331.8817615509033, 312.6790523529053, 330.2266597747803, 313.85254859924316, 303.0574321746826, 309.31878089904785, 314.03040885925293, 307.1613311767578, 317.14510917663574, 315.53173065185547
    , 346.70567512512207, 318.39847564697266, 345.31188011169434, 312.4120235443115, 317.6300525665283, 313.1091594696045, 311.90967559814453, 312.9432201385498, 317.0444965362549, 328.16243171691895
    , 316.2844181060791, 323.5750198364258, 321.92134857177734, 311.2916946411133, 307.30581283569336, 315.91010093688965, 331.4542770385742, 307.178258895874, 316.8618679046631, 307.830810546875
    , 318.6826705932617, 334.6741199493408, 307.50036239624023, 313.5535717010498, 306.28466606140137, 322.0992088317871, 310.8499050140381, 312.79802322387695, 324.4340419769287, 309.722900390625
    , 354.62284088134766, 316.3599967956543, 347.52511978149414, 317.547082901001, 320.3754425048828, 313.8546943664551, 324.17988777160645, 313.3211135864258, 310.57238578796387, 313.47131729125977   
    ]



    awgBB = Average(list_bertBase)
    awgRN = Average(list_Resnet)
    awgI3 = Average(list_inception)

    perR_BB = awgBB/6
    perR_RN = awgRN/6
    perR_I3 = awgI3/6
    print('(Request aware PipeSwitch) Average time for BB is : ' + str(awgBB) +' with a perRun of: ' + str(perR_BB))
    print('Request aware PipeSwitch) Average time for RN is : ' + str(awgRN) +' with a perRun of: ' + str(perR_RN))
    print('Request aware PipeSwitch) Average time for I3 is : ' + str(awgI3) +' with a perRun of: ' + str(perR_I3))

    #Using NewWorker
    list_bertBase_ps =[671.208381652832, 332.9291343688965, 332.7045440673828, 333.7075710296631, 333.7242603302002, 334.9635601043701, 334.11288261413574, 333.52208137512207, 333.61101150512695, 335.6022834777832, 334.11359786987305, 334.6869945526123, 337.47172355651855, 336.0929489135742, 336.1384868621826, 335.30139923095703, 340.90518951416016, 335.22558212280273, 335.3002071380615, 335.86835861206055, 339.25533294677734, 336.0905647277832, 338.8097286224365, 335.7698917388916, 335.7577323913574, 335.66808700561523, 335.02912521362305, 337.22901344299316, 333.8637351989746, 334.1064453125, 334.34081077575684, 333.96244049072266, 336.7798328399658, 332.81683921813965, 333.85586738586426, 335.3581428527832, 335.31808853149414, 335.95824241638184, 336.40217781066895, 332.0019245147705, 332.52906799316406, 332.36193656921387, 332.3173522949219, 334.2881202697754, 332.8824043273926, 333.13918113708496, 332.3342800140381, 332.2179317474365, 332.86428451538086, 332.96680450439453]
    list_resnet_ps = [552.3386001586914, 464.32042121887207, 482.35368728637695, 456.18200302124023, 453.4945487976074, 453.7985324859619, 454.87236976623535, 454.14161682128906, 445.9824562072754, 448.760986328125, 446.03514671325684, 440.4644966125488, 448.589563369751, 462.2783660888672, 454.15520668029785, 448.91357421875, 448.44627380371094, 448.5805034637451, 464.0772342681885, 461.101770401001, 457.81493186950684, 457.25083351135254, 449.7082233428955, 451.57456398010254, 464.82110023498535, 469.65575218200684, 456.7422866821289, 459.90991592407227, 455.9593200683594, 457.63659477233887, 451.4741897583008, 454.6639919281006, 453.11975479125977, 453.95421981811523, 451.85256004333496, 451.9503116607666, 451.88236236572266, 458.07647705078125, 460.61134338378906, 465.92116355895996, 464.57552909851074, 454.39743995666504, 462.4326229095459, 463.75060081481934, 479.02512550354004, 452.52466201782227, 452.50535011291504, 455.6436538696289, 452.9120922088623, 454.78034019470215]
    list_inception_ps = [402.493953704834, 381.12616539001465, 375.7641315460205, 378.2517910003662, 391.94297790527344, 386.90829277038574, 385.7874870300293, 381.32500648498535, 393.53036880493164, 378.10635566711426, 380.3706169128418, 384.14645195007324, 380.74421882629395, 387.80784606933594, 387.76302337646484, 380.9518814086914, 377.6700496673584, 388.8838291168213, 380.15198707580566, 377.122163772583, 384.7966194152832, 388.4015083312988, 400.6335735321045, 385.47658920288086, 390.49530029296875, 369.3656921386719, 372.62940406799316, 381.056547164917, 377.8676986694336, 365.68522453308105, 382.51233100891113, 368.757963180542, 373.71087074279785, 371.86622619628906, 373.884916305542, 368.1912422180176, 377.2470951080322, 378.262996673584, 378.05938720703125, 394.0098285675049, 389.8491859436035, 399.8243808746338, 420.92061042785645, 391.01552963256836, 379.08482551574707, 387.58087158203125, 380.89680671691895, 389.8742198944092, 376.16944313049316, 374.2823600769043]

    
    awgBB_ps = Average(list_bertBase_ps)
    awgRN_ps = Average(list_resnet_ps)
    awgI3_ps = Average(list_inception_ps)
  

    perR_BB_ps = awgBB_ps/6
    perR_RN_ps = awgRN_ps/6
    perR_I3_ps = awgI3_ps/6
    
    print('(PipeSwitch) Average time for BB is : ' + str(awgBB_ps) +' with a perRun of: ' + str(perR_BB_ps))
    print('(PipeSwitch) Average time for RN is : ' + str(awgRN_ps) +' with a perRun of: ' + str(perR_RN_ps))
    print('(PipeSwitch) Average time for I3 is : ' + str(awgI3_ps) +' with a perRun of: ' + str(perR_I3_ps))



    print("-------------------------------------------------")
    print("Complex sequences 50 runs each (multiple requests in each run): ")
    #bert_base;bert_base;bert_base;bert_base;inception_v3;resnet152;resnet152;resnet152;resnet152;resnet152;resnext101_32x8d;resnext101_32x8d
    #12 requests total
    awgComplex1 = [1593.6133861541748, 1110.0578308105469, 1094.5358276367188, 1106.58860206604, 1093.0588245391846, 1118.8020706176758, 1096.073865890503, 1074.2545127868652, 1083.885908126831, 1062.180995941162, 1081.2010765075684, 1078.545331954956, 1072.73530960083, 1062.0603561401367, 1075.347661972046, 1075.4215717315674, 1066.889762878418, 1060.7233047485352, 1066.8301582336426, 1063.082218170166, 1064.096212387085, 1068.1326389312744, 1049.9398708343506, 1058.3930015563965, 1056.0436248779297, 1062.2808933258057, 1074.5394229888916, 1068.406105041504, 1061.220407485962, 1061.8090629577637, 1059.278964996338, 1059.5266819000244, 1060.018539428711, 1070.969820022583, 1078.4566402435303, 1064.41330909729, 1060.1513385772705, 1060.3327751159668, 1059.3538284301758, 1064.4068717956543, 1058.5741996765137, 1070.7712173461914, 1067.814826965332, 1074.1899013519287, 1060.654640197754, 1075.0384330749512, 1058.2447052001953, 1056.8451881408691, 1055.8538436889648, 1060.0154399871826]
    
    awgComplex2 = [1168.379783630371, 661.121129989624, 677.9458522796631, 656.475305557251, 680.5388927459717, 658.6616039276123, 659.0001583099365, 659.8625183105469, 656.1570167541504, 671.6067790985107, 673.2852458953857, 667.9766178131104, 662.8804206848145, 662.2028350830078, 668.4911251068115, 660.1300239562988, 662.2536182403564, 651.8270969390869, 662.4047756195068, 658.8108539581299, 661.5946292877197, 664.2217636108398, 680.5558204650879, 666.8620109558105, 668.8363552093506, 661.2777709960938, 662.1637344360352, 658.571720123291, 665.7445430755615, 658.4174633026123, 675.2157211303711, 667.7124500274658, 685.4259967803955, 669.5303916931152, 662.7118587493896, 656.9883823394775, 666.1393642425537, 664.9978160858154, 660.5160236358643, 659.6651077270508, 678.0273914337158, 664.8268699645996, 661.552906036377, 666.8601036071777, 662.3344421386719, 659.6112251281738, 660.102128982544, 658.4076881408691, 665.4179096221924, 657.2983264923096]
    
    awgComplex3 = []

def Average(lst):
    return sum(lst) / len(lst)    
if __name__ == '__main__':
    main()
