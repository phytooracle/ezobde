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
def create_labels(data, data_loaded):
  
    args = get_args()
    print('>>> Creating XML label files.')
    for i in range(len(data)):
        try:
            file_name = data[i]['External ID'].replace('.png', '.txt')
            name = data[i]['External ID']
            out_name = name.replace('.png', '.xml')

            if name in test:
                file_type = data_loaded['outputs']['test_outdir']

            elif name in train: 
                file_type = data_loaded['outputs']['train_outdir']

            else:
                file_type = data_loaded['outputs']['validation_outdir']

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
    dataset = core.Dataset(os.path.join(os.getcwd(), data_loaded['outputs']['train_outdir']))
    loader = core.DataLoader(dataset, batch_size=data_loaded['training_parameters']['batch_size'], shuffle=data_loaded['training_parameters']['shuffle'])
    val_dataset = core.Dataset(os.path.join(os.getcwd(), data_loaded['outputs']['validation_outdir']))
    
    # Define model
    model = core.Model(data_loaded['training_parameters']['classes'])

    # Train model
    losses = model.fit(loader, val_dataset, epochs=data_loaded['training_parameters']['epochs'], learning_rate=data_loaded['training_parameters']['learning_rate'], verbose=data_loaded['training_parameters']['verbose'])
    plt.plot(losses)
    plt.savefig(data_loaded['outputs']['plot_outfile'])
    
    print('>>> Training complete.')

    # Save model
    print('>>> Saving model.')
    model.save(data_loaded['outputs']['model_outfile'])
    print(f">>> Model saved. See {data_loaded['outputs']['model_outfile']}")


# --------------------------------------------------
def main():
    """Make a jazz noise here"""

    args = get_args()
    data_loaded = args.yaml
    
    # Define API key & project ID
    api_key = data_loaded['credentials']['api_key']
    project_id = data_loaded['credentials']['project_id']
    project, labels, labels_annotation = get_labelbox_data(api_key, project_id)
    
    # Create train/test/validation split
    global train, val, test
    train, val, test, img_dict = split_data(labels)

    # Download image data
    download_set (data_loaded['outputs']['train_outdir'], train, img_dict)
    download_set (data_loaded['outputs']['validation_outdir'], val, img_dict)
    download_set (data_loaded['outputs']['test_outdir'], test, img_dict)
    
    # Create labels
    create_labels(labels, data_loaded)

    # Train model 
    if data_loaded['training_parameters']['train_model']:
        train_model(data_loaded)


# --------------------------------------------------
if __name__ == '__main__':
    main()
