#!/usr/bin/env python3
"""
Author : Emmanuel Gonzalez
Date   : 2022-04-08
Purpose: EZOBDE | EaZy Object Detection
"""

import argparse
import os
import sys
import subprocess as sp
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import labelbox
from labelbox import Client, OntologyBuilder
from labelbox.data.annotation_types import Geometry
from getpass import getpass
from PIL import Image
import random
import cv2
from pascal_voc_writer import Writer
from detecto.core import Dataset
from detecto import core, utils, visualize
import glob
from xml.dom import minidom
import yaml
from torchvision import transforms
from detecto.utils import normalize_transform
import xml.etree.ElementTree as ET


# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='EZOBDE | EaZy Object Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-y',
                        '--yaml',
                        help='YAML file containing arguments',
                        metavar='str',
                        type=str,
                        default='config.yaml')

    args = parser.parse_args()

    args.yaml = yaml.safe_load(open(args.yaml, 'r'))

    return args


# --------------------------------------------------
def get_labels(api_key, project_id):

  # Enter your Labelbox API key here
  LB_API_KEY = api_key

  # Create Labelbox client
  lb = labelbox.Client(api_key=LB_API_KEY)
  
  # Get project by ID
  project = lb.get_project(project_id)
  
  # Export image and text data as an annotation generator:
  labels_annotation = project.label_generator()

  # Export labels as a json:
  labels = project.export_labels(download = True)

  return project, labels, labels_annotation


# --------------------------------------------------
def download_set(work_path, set_list, img_dict):
    
    test_type = work_path.split('/')[-1]

    if not os.path.isdir(work_path):
        os.makedirs(work_path)
    else: 
        print(f'{test_type.capitalize()} set already exists.')

    print('>>> Downloading data.')
    for item in set_list: 

        url = img_dict.get(item)

        if not os.path.isfile(f'{os.path.join(os.getcwd(), work_path, item)}'):

            print(f'>>> Downloading {item}.')
            sp.call(f'wget "{url}" -O {os.path.join(work_path, item)}', shell=True) 
    print('>>> Download complete.')


# --------------------------------------------------
def split_data(labels): 

  img_list = [item['Labeled Data'] for item in labels if item['Skipped']==False]
  name_list = [item['External ID'] for item in labels if item['Skipped']==False]
  id_list = [item['ID'] for item in labels if item['Skipped']==False]
  img_dict = dict(zip(name_list, img_list))
  label_dict = dict(zip(name_list, id_list))

  train, val, test = np.split(name_list, [int(.8*len(name_list)), int(.9*len(name_list))])
  return train, val, test, img_dict


# --------------------------------------------------
def create_labels(data, data_loaded, file_extension):
  
    args = get_args()
    print('>>> Creating XML label files.')
    for i in range(len(data)):
        try:
            file_name = data[i]['External ID'].replace(file_extension, '.txt')
            name = data[i]['External ID']
            out_name = name.replace(file_extension, '.xml')

            if name in test:
                file_type = os.path.join(data_loaded['data']['root_dir'], data_loaded['outputs']['test_outdir'])

            elif name in train: 
                file_type = os.path.join(data_loaded['data']['root_dir'], data_loaded['outputs']['train_outdir'])

            else:
                file_type = os.path.join(data_loaded['data']['root_dir'], data_loaded['outputs']['validation_outdir'])

            # print(os.path.join(file_type, name))
            if os.path.isfile(os.path.join(file_type, name)):
                if not os.path.isfile(os.path.join(file_type, out_name)):
                    print(f'>>> Creating {out_name}.')
                    img = cv2.imread(os.path.join(file_type, name))

                    h, w, _ = img.shape
                    label_list, x, y = [], [], []
                    for a in range(len(data[i]['Label']['objects'])):

                        points = data[i]['Label']['objects'][a]['bbox']
                        label = data[i]['Label']['objects'][a]['value']

                        label_list.append(label)
                        x.append([points['left'], (points['left'] + points['width'])])
                        y.append([points['top'], (points['top'] + points['height'])])

                    final = list(zip(label_list, x, y))
                    if not final:
                        print('>>> Empty')

                    name = os.path.join(file_type, name)
                    writer = Writer(name, w, h)
                    for item in final:

                        min_x, max_x = item[1]
                        min_y, max_y = item[2]
                        writer.addObject(item[0], min_x, min_y, max_x, max_y)
                        writer.save(os.path.join(file_type, out_name))
        
        except:
            pass

    print('>>> Done creating labels.')


# --------------------------------------------------
def get_labelbox_data(api_key, project_id):

    project, labels, labels_annotation = get_labels(api_key, project_id)

    return project, labels, labels_annotation


# --------------------------------------------------
def train_model(data_loaded):
    
    print('>>> Model training')

    # Define datasets
    if data_loaded['training_parameters']['transforms']:
        print(type(data_loaded['training_parameters']['transforms']))
        custom_transforms = transforms.Compose([exec(item) for item in data_loaded['training_parameters']['transforms']])
        dataset = core.Dataset(os.path.join(os.getcwd(), data_loaded['data']['root_dir'], data_loaded['outputs']['train_outdir']), transform=custom_transforms)

    else:

        dataset = core.Dataset(os.path.join(os.getcwd(), data_loaded['data']['root_dir'], data_loaded['outputs']['train_outdir']))

    loader = core.DataLoader(dataset, batch_size=data_loaded['training_parameters']['batch_size'], shuffle=data_loaded['training_parameters']['shuffle'])
    val_dataset = core.Dataset(os.path.join(os.getcwd(), data_loaded['data']['root_dir'], data_loaded['outputs']['validation_outdir']))
    
    # Define model
    model = core.Model(data_loaded['training_parameters']['classes'])

    # Train model
    losses = model.fit(loader, val_dataset, epochs=data_loaded['training_parameters']['epochs'], learning_rate=data_loaded['training_parameters']['learning_rate'], verbose=data_loaded['training_parameters']['verbose'])
    plt.plot(losses)
    plt.savefig(os.path.join(data_loaded['data']['root_dir'], data_loaded['outputs']['plot_outfile']))
    
    print('>>> Training complete.')

    # Save model
    print('>>> Saving model.')
    model.save(os.path.join(data_loaded['data']['root_dir'], data_loaded['outputs']['model_outfile']))
    print(f">>> Model saved.")


# --------------------------------------------------
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def assess_model_performance(model_path, image_set, class_list, csv_outfile, date_string, save_predictions, file_extension):

    detect_dict = {}
    iou_dict = {}
    gt_num = []
    img_list = []
    model = core.Model.load(model_path, class_list)

    if save_predictions:

        if not os.path.isdir(save_predictions):
            os.makedirs(save_predictions)

    for img in glob.glob(os.path.join(image_set, ''.join(['*', file_extension]))):
        
        try:
            cnt = 0
            image = utils.read_image(img)
            predictions = model.predict(image)
            labels, boxes, scores = predictions
            a_img = cv2.imread(img)
            a_img = cv2.cvtColor(a_img, cv2.COLOR_BGR2RGB)
            copy = a_img.copy()

            xml = img.replace(file_extension, '.xml')
            mydoc = minidom.parse(xml)
            items = mydoc.getElementsByTagName('object')
            tree = ET.parse(xml)
            root = tree.getroot()
            gt = len([roi for roi in root.iter('object')])
            gt_num.append(gt)
            img_list.append(img)

            iou_list = []
            for i, box in enumerate(boxes):

                min_x, min_y, max_x, max_y = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                ml = [min_y, min_x, max_y, max_x]
                start_point = (min_x, max_y)
                end_point = (max_x, min_y)
                color = (255, 0, 0) 
                thickness = 6
                cv2.rectangle(a_img, start_point, end_point, color, thickness)

                result_list = []
                for roi in root.iter('object'):
                    file_name = root.find('filename').text
                    ymin, xmin, ymax, xmax = None, None, None, None 

                    ymin = int(roi.find("bndbox/ymin").text)
                    xmin = int(roi.find("bndbox/xmin").text)
                    ymax = int(roi.find("bndbox/ymax").text)
                    xmax = int(roi.find("bndbox/xmax").text)
                    gt = [ymin, xmin, ymax, xmax]
                    start_point = (xmin, ymax)
                    end_point = (xmax, ymin)
                    color = (0, 0, 255) 
                    thickness = 6
                    cv2.rectangle(a_img, start_point, end_point, color, thickness)
                    
                    iou = bb_intersection_over_union(gt, ml)
                    result_list.append(iou)

                final_iou = max(result_list)
                iou_list.append(final_iou)

            if save_predictions:
                print(f'>>> Saving predictions for {img}')
                cv2.imwrite(os.path.join(save_predictions, os.path.basename(img.replace(file_extension, f'_prediction{file_extension}'))), a_img)

            iou_dict[file_name] = {
                'iou': iou_list
            }
            
        except:
            pass

    df = pd.DataFrame.from_dict(iou_dict, orient='index').explode('iou')
    df['iou'] = df['iou'].astype(float)
    # df = df.groupby(by=df.index).mean()
    df = df.reset_index()

    if date_string:
        df['date'] = df['index'].str.split('_', expand=True)[0]
        df = df.sort_values('date')

    df.to_csv(csv_outfile, index=False)

    return df

# --------------------------------------------------
def main():
    """Make a jazz noise here"""

    args = get_args()
    data_loaded = args.yaml
    
    # Download image data
    if data_loaded['outputs']['download_from_labelbox']:

        # Define API key & project ID
        api_key = data_loaded['credentials']['api_key']
        project_id = data_loaded['credentials']['project_id']
        project, labels, labels_annotation = get_labelbox_data(api_key, project_id)
        
        # Create train/test/validation split
        global train, val, test
        train, val, test, img_dict = split_data(labels)

        # Download data from LabelBox
        download_set(os.path.join(data_loaded['data']['root_dir'], data_loaded['outputs']['train_outdir']), train, img_dict)
        download_set(os.path.join(data_loaded['data']['root_dir'], data_loaded['outputs']['validation_outdir']), val, img_dict)
        download_set(os.path.join(data_loaded['data']['root_dir'], data_loaded['outputs']['test_outdir']), test, img_dict)
    
        # Create labels
        create_labels(labels, data_loaded, file_extension=data_loaded['data']['file_extension'])

    # Train model 
    if data_loaded['training_parameters']['train_model']:
        if not os.path.isfile(os.path.join(data_loaded['data']['root_dir'], data_loaded['outputs']['model_outfile'])):
            train_model(data_loaded)
        else:
            print('Previously trained model found, loading it.')

    if data_loaded['performance_parameters']['assess_performance']:
        
        assess_model_performance(model_path = os.path.join(data_loaded['data']['root_dir'], data_loaded['outputs']['model_outfile']),
                                 file_extension = data_loaded['data']['file_extension'],
                                 date_string = data_loaded['data']['date_string'],
                                 save_predictions = os.path.join(data_loaded['data']['root_dir'], data_loaded['performance_parameters']['save_predictions']),
                                 image_set = os.path.join(data_loaded['data']['root_dir'], data_loaded['performance_parameters']['test_directory']),
                                 class_list = data_loaded['training_parameters']['classes'],
                                 csv_outfile = os.path.join(data_loaded['data']['root_dir'], data_loaded['performance_parameters']['csv_outfile']))


# --------------------------------------------------
if __name__ == '__main__':
    main()
