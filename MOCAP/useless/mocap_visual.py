import json
import csv

# {
#     "Frame:"1,
#     "keypoints":{
#         "type":"Bone",
#         "Name":"Aliprandi_Girardi:Hip",
#         "ID":1,
# 	“Rotation”:{
# 		"X":-0.225563,
#         	"Y":0.960407,
#         	"Z":0.049586
# 		“W”:0.4334
# 	},
# 	“Position”:{
#         	"X":-0.225563,
#         	"Y":0.960407,
#         	"Z":0.049586
# 	}
#     }
# }

array=[]
frames=[]

with open('groundTruth.csv', 'r') as file:
    reader = csv.reader(file)

    for i,row in enumerate(reader):

        array_element=[]
        for j,element in enumerate(row):
            array_element.append(element)
        array.append(array_element)

        struct={
                "Frame":1,
                "keypoints":[]
            }

i=4
old_ID="ID"
j=2
while (j<(len(array[i])-3)):
    struct={
            "Frame":i-4,
            "keypoints":[]
        }
    joint = {
        "type":array[i-2][j],
        "Name":array[i-1][j],
        "ID":array[i][j],
        "Rotation":[],
        "Position":[]
    }
    for l in range(i+1,len(array)):
        have_I_read_rotation=False;
        if (array[i+1][j]=="Rotation"):
            have_I_read_rotation=True
            rotation={
                "X":array[l+2][j],
                "Y":array[l+2][j+1],
                "Z":array[l+2][j+2],
                "W":array[l+2][j+3]
            }
            joint["Rotation"]=rotation
        elif (array[i+1][j]=="Position"):
            position={
                "X":array[l+2][j],
                "Y":array[l+2][j+1],
                "Z":array[l+2][j+2],
            }
            joint["Position"]=position
        if (have_I_read_rotation):
            j+=4
        else:
            break;
    j+=3
    struct["keypoints"].append(joint)
    frames.append(struct)


print(frames)
