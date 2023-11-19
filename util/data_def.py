SEG_LABELS_LIST_v1 = [
    {"id": -1, "name": "void",       "rgb_values": [0,   0,    0]},
    {"id": 0,  "name": "building",   "rgb_values": [128, 0,    0]},
    {"id": 1,  "name": "grass",      "rgb_values": [0,   128,  0]},
    {"id": 2,  "name": "tree",       "rgb_values": [128, 128,  0]},
    {"id": 3,  "name": "cow",        "rgb_values": [0,   0,    128]},
    {"id": 4,  "name": "sky",        "rgb_values": [128, 128,  128]},
    {"id": 5,  "name": "airplane",   "rgb_values": [192, 0,    0]},
    {"id": 6, "name": "face",       "rgb_values": [192, 128,  0]},
    {"id": 7, "name": "car",        "rgb_values": [64,  0,    128]},
    {"id": 8, "name": "bicycle",    "rgb_values": [192, 0,    128]},
    {"id": 9, "name": "horse",       "rgb_values": [128,   0,    128]},
    {"id": 10, "name": "water",       "rgb_values": [64,   128,    0]},
    {"id": 11, "name": "mountain",       "rgb_values": [64,   0,    0]},
    {"id": 12, "name": "sheep",       "rgb_values": [0,   128,    128]}]

rgb_2_label = {}
label_2_rgb = {}
for i in SEG_LABELS_LIST_v1:
    rgb_2_label[tuple(i['rgb_values'])] = i['id']
    label_2_rgb[i['id']] = tuple(i['rgb_values'])


current_directory = 'c:/Users/steve/Desktop/CMSC828I/Hw1/data'
msrc_directory = current_directory + '/MSRC_ObjCategImageDatabase_v1'
save_data_path = current_directory + '/Train_dataset/'
save_train_txt_path = current_directory + '/text_file/'
txt_file =  save_train_txt_path + "/train.txt"
img_path = save_data_path
model_path = current_directory + '/save_model/seg_network_9.pt'
save_model = current_directory + '/save_model/'    