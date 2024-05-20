import yarp
import cv2
import json
import math
import glob
import yarp
import PIL
import sys
import io
import logging
import torch
import argparse, os
import torchvision
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import datasets, transforms
from PIL import Image
from PIL import ImageShow
from scipy.misc import imresize
from model import ModelSpatial
from utils import imutils, evaluation
from config import *
from functions.config_vt import *
from functions.utilities_vt import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_weights', type=str, help='model weights', default='/projects/online-visual-target-detection/model_demo.pt')
# parser.add_argument('--image_dir', type=str, help='images', default=FRAMES_DIR)
# parser.add_argument('--head_dir', type=str, help='head bounding boxes', default=JSON_FILES)
parser.add_argument('--vis_mode', type=str, help='heatmap or arrow', default='arrow')
parser.add_argument('--out_threshold', type=int, help='out-of-frame target dicision threshold', default=200)
args = parser.parse_args()

if __name__ == '__main__':
    dataset_folder = os.path.join(os.getcwd(), "hsp_dataset/test")
    # for each split
    list_splits = [name for name in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, name))]
    list_splits.sort()
    #list_splits = ['split4']
    for split in list_splits:
        args.model_weights = '/projects/vtd-comparison-maria/VTD_retrain/src/logs/%s/epoch_03_weights.pt' % split
        print('Model loaded: %s' % args.model_weights)
        split_folder = os.path.join(dataset_folder, split)
        list_participants = [name for name in os.listdir(split_folder) if os.path.isdir(os.path.join(split_folder, name))]
        #list_participants = ['p00']
        list_participants.sort()
        for participant in list_participants:
            participant_folder = os.path.join(split_folder, participant)
            print('Processing folder %s' % participant_folder)

            output_file = open(participant_folder + '/results_finetuned.txt', "w+")

            imgs_folder = os.path.join(participant_folder, 'board_images_human')
            json_folder = os.path.join(participant_folder, 'board_data_openpose')
            #out_folder = os.path.join(participant_folder, 'output_images')
            #if not os.path.exists(out_folder):
            #    os.makedirs(out_folder)

            imgs = list(filter(lambda x: '.jpg' in x, os.listdir(imgs_folder)))
            imgs = [img.replace('.jpg', '') for img in imgs]
            imgs.sort()
            for img in imgs:
                img_file = os.path.join(imgs_folder, img + '.jpg')
                json_file = os.path.join(json_folder, img + '_keypoints.json')
                poses, conf_poses, faces, conf_faces = read_openpose_from_json(json_file)
                pil_image = Image.open(img_file)
                pil_image = pil_image.convert('RGB')

                if poses:
                    min_x, min_y, max_x, max_y = get_openpose_bbox(poses)

                    column_names = ['left', 'top', 'right', 'bottom']
                    line_to_write = [[min_x, min_y, max_x, max_y]]
                    df = pd.DataFrame(line_to_write, columns=column_names)

                    df['left'] -= (df['right'] - df['left']) * 0.1
                    df['right'] += (df['right'] - df['left']) * 0.1
                    df['top'] -= (df['bottom'] - df['top']) * 0.1
                    df['bottom'] += (df['bottom'] - df['top']) * 0.1

                    # Transforming images
                    def get_transform():
                        transform_list = []
                        transform_list.append(transforms.Resize((input_resolution, input_resolution)))
                        transform_list.append(transforms.ToTensor())
                        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
                        return transforms.Compose(transform_list)

                     # set up data transformation
                    test_transforms = get_transform()

                    model = ModelSpatial()
                    model_dict = model.state_dict()
                    pretrained_dict = torch.load(args.model_weights)
                    pretrained_dict = pretrained_dict['model']
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict)

                    model.cuda()
                    model.train(False)

                    with torch.no_grad():
                        for i in df.index:
                            frame_raw = pil_image

                            width, height = frame_raw.size

                            head_box = [df.loc[i,'left'], df.loc[i,'top'], df.loc[i,'right'], df.loc[i,'bottom']]

                            head = frame_raw.crop((head_box)) # head crop

                            head = test_transforms(head) # transform inputs
                            frame = test_transforms(frame_raw)
                            head_channel = imutils.get_head_box_channel(head_box[0], head_box[1], head_box[2], head_box[3], width, height,
                                                                        resolution=input_resolution).unsqueeze(0)

                            head = head.unsqueeze(0).cuda()
                            frame = frame.unsqueeze(0).cuda()
                            head_channel = head_channel.unsqueeze(0).cuda()

                            # forward pass
                            raw_hm, _, inout = model(frame, head_channel, head)

                            # heatmap modulation
                            raw_hm = raw_hm.cpu().detach().numpy() * 255
                            raw_hm = raw_hm.squeeze()
                            inout = inout.cpu().detach().numpy()
                            inout = 1 / (1 + np.exp(-inout))
                            inout = (1 - inout) * 255
                            norm_map = imresize(raw_hm, (height, width)) - inout

                            # Visualization
                            # Draw the raw_frame and the bbox
                            start_point = (int(head_box[0]), int(head_box[1]))
                            end_point = (int(head_box[2]), int(head_box[3]))
                            img_bbox = cv2.rectangle(np.asarray(frame_raw),start_point,end_point, (0, 255, 0),2)
                                
                            # The arrow mode
                            if args.vis_mode == 'arrow':
                                # in-frame gaze
                                #if inout < args.out_threshold:
                                pred_x, pred_y = evaluation.argmax_pts(raw_hm)
                                norm_p = [pred_x/output_resolution, pred_y/output_resolution]
                                print('image %s: pred [%d %d], norm_pred [%f %f], scaled [%d %d]' % (img, pred_x, pred_y, norm_p[0], norm_p[1], int(norm_p[0]*width), int(norm_p[1]*height)))
                                output_file.write("%s %d %d %f %f %d %d \n" % (img, pred_x, pred_y, norm_p[0], norm_p[1], int(norm_p[0]*width), int(norm_p[1]*height)))
                                #circs = cv2.circle(img_bbox, (int(norm_p[0]*width), int(norm_p[1]*height)),  int(height/50.0), (35, 225, 35), -1)
                                #cv2.imwrite(out_folder + '/' + img + '.jpg', cv2.cvtColor(circs, cv2.COLOR_BGR2RGB))
                else:
                    print('Cannot read json file for the image %s' % img_file)

