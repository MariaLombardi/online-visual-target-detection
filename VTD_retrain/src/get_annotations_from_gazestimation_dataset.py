#!/usr/bin/python3
import os, csv
import pandas as pd
import numpy as np
from utilities import read_openpose_from_json, get_openpose_bbox, read_openpose_data
import yarp

root_dir = os.getcwd()
dataset_dir = os.path.join(root_dir, '../../src/hsp_dataset/train')
cameras = ['icubCamera', 'realsense']

for camera in cameras:
    annotations_path = os.path.join(dataset_dir, 'board_annotations_%s.txt' % camera)
    annotations_file = open(annotations_path, "r")
    annotations_contents = annotations_file.readlines()
    annotations_contents = [x.strip() for x in annotations_contents]
    annotations = [x.split(" ") for x in annotations_contents]
    df_ground_truth = pd.DataFrame(columns=['participant', 'image', 'face_center', 'gaze_vector'])
    for it in range(0, len(annotations)):
        annotation = annotations[it]
        annotation_split = annotation[0].split("/")
        participant = annotation_split[1]
        sample = (annotation_split[-1]).replace('.jpg', '')
        centroid_pix = np.array(list(map(float, annotation[5:7])))
        target_pix = np.array(list(map(float, annotation[15:17])))
        gaze_target_pix = target_pix - centroid_pix
        gaze_pix = (gaze_target_pix / np.linalg.norm(gaze_target_pix)) * 50

        df_ground_truth = df_ground_truth.append({ 'participant': participant,
                    'image': sample,
                    'face_center': centroid_pix,
                    'gaze_vector': gaze_pix
        }, ignore_index=True)

    list_splits = [name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, name))]
    list_splits.sort()
    for split in list_splits:
        split_folder = os.path.join(dataset_dir, split)

        annotation_file = open(split_folder + '/annotations_%s.csv' % camera, 'w')
        csv_writer = csv.writer(annotation_file)
        csv_header = ['path', 'eye_x', 'eye_y', 'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'inout']
        csv_writer.writerow(csv_header)

        list_participants = [name for name in os.listdir(split_folder) if os.path.isdir(os.path.join(split_folder, name))]
        # list_participants = ['p00']
        list_participants.sort()
        for participant in list_participants:
            participant_folder = os.path.join(split_folder, participant, camera)
            print('Processing folder %s' % participant_folder)
            imgs_folder = os.path.join(participant_folder, 'board_images_human')
            json_folder = os.path.join(participant_folder, 'board_data_openpose')
            imgs = list(filter(lambda x: '.jpg' in x, os.listdir(imgs_folder)))
            imgs = [img.replace('.jpg', '') for img in imgs]
            imgs.sort()
            for img in imgs:
                img_file = os.path.join(imgs_folder, img + '.jpg')
                if camera == 'icubCamera':
                    json_file = os.path.join(json_folder, img + '_keypoints.json')
                    poses, conf_poses, faces, conf_faces = read_openpose_from_json(json_file)
                else:
                    pose_file = os.path.join(json_folder, 'data.log')
                    openpose_file_contents = (open(pose_file, "r")).readlines()
                    openpose_contents = [x.strip() for x in openpose_file_contents]

                    openpose_row = openpose_contents[int(sample)]
                    openpose_string = openpose_row[openpose_row.find("((("):]
                    openpose_data = yarp.Bottle(openpose_string)
                    poses, conf_poses, faces, conf_faces = read_openpose_data(openpose_data)

                if poses:
                    min_x, min_y, max_x, max_y = get_openpose_bbox(poses)
                    row = df_ground_truth[(df_ground_truth['participant'] == participant) & (df_ground_truth['image'] == img)]
                    eye = (row['face_center'].values.tolist())[0]
                    gaze = (row['gaze_vector'].values.tolist())[0]
                    csv_row = [participant + '/' + camera + '/board_images_human/' + img + '.jpg',
                               eye[0],
                               eye[1],
                               gaze[0],
                               gaze[1],
                               min_x,
                               min_y,
                               max_x,
                               max_y,
                               1]
                    csv_writer.writerow(csv_row)
                else:
                    print('Cannot read json file for the image %s' % img_file)

        annotation_file.close()



