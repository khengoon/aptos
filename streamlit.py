import streamlit as st
import urllib.request
import os
import sys
import random
import math
import subprocess
from glob import glob
from collections import OrderedDict
import numpy as np
# import cv2
from skimage import measure
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from matplotlib import pyplot as plt
import scipy.ndimage as ndi
from PIL import Image, ImageOps
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
from sklearn.linear_model import LinearRegression

import torch
from torch import nn
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
from torch.optim.optimizer import Optimizer, required
from torch.autograd import Variable

from utils import lottie_eye
from streamlit_lottie import st_lottie
from model import *
from pathlib import Path

# https://www.kaggle.com/stormdiv/nctu-cs-t0828-final-aptos-2019-0856152/notebook
# st.set_page_config(layout='wide')

################################################################################

st_lottie(lottie_eye, height=200)

st.title('APTOS 2019 Blindness Detection')

st.markdown('Disclaimer: This is a project by Low Kheng Oon. This application is not production ready. Use at your own discretion')

st.header("Diabetic Retinopathy")

st.markdown("Imagine being able to detect blindness before it happened.")

st.markdown("Diabetic retinopathy (DR), also known as diabetic eye disease, is a medical condition in which damage occurs to the retina due to diabetes mellitus. It is a leading cause of blindness. Diabetic retinopathy affects up to 80 percent of those who have had diabetes for 20 years or more. Diabetic retinopathy often has no early warning signs. **Retinal (fundus) photography with manual interpretation is a widely accepted screening tool for diabetic retinopathy**, with performance that can exceed that of in-person dilated eye examinations. ")

st.markdown("The below figure shows an example of a healthy patient and a patient with diabetic retinopathy as viewed by fundus photography ([source](https://www.biorxiv.org/content/biorxiv/early/2018/06/19/225508.full.pdf)):")

col1, col2, col3 = st.columns([1,6,1])

with col1:
    st.write("")

with col2:
    # urllib.request.urlretrieve("https://www.researchgate.net/profile/Srinivasa-Rao-Avanapu/publication/282609747/figure/fig2/AS:281548759814145@1444137863086/Difference-between-Normal-Retina-and-Diabetic-Retinopathy_W640.jpg", "intro_img")
    intro_img = Image.open('Difference-between-Normal-Retina-and-Diabetic-Retinopathy_W640.jpg')
    # intro_img = Image.open("intro_img")
    st.image(intro_img)


with col3:
    st.write("")

st.markdown("An automated tool for grading severity of diabetic retinopathy would be very useful for accerelating detection and treatment. Recently, there have been a number of attempts to utilize deep learning to diagnose DR and automatically grade diabetic retinopathy. This includes a [work by Google](https://ai.googleblog.com/2016/11/deep-learning-for-detection-of-diabetic.html) and Even one deep-learning based system is [FDA approved](https://www.fda.gov/NewsEvents/Newsroom/PressAnnouncements/ucm604357.htm).")

st.subheader("A look at the data:")

st.text("Data description from the competition:")

st.text("You are provided with a large set of high-resolution retina images taken under a variety of imaging conditions. A left and right field is provided for every subject. >Images are labeled with a subject id as well as either left or right (e.g. 1_left.jpeg is the left eye of patient id 1).")

st.text("A clinician has rated the presence of diabetic retinopathy in each image on a scale of 0 to 4, according to the following scale:")

st.caption("0 - No DR")

st.caption("1 - Mild")

st.caption("2 - Moderate")

st.caption("3 - Severe")

st.caption("4 - Proliferative DR")

st.text("Your task is to create an automated analysis system capable of assigning a score based on this scale.")

st.caption("Like any real-world data set, you will encounter noise in both the images and labels. Images may contain artifacts, be out of focus, underexposed, or overexposed. A major aim of this competition is to develop robust algorithms that can function in the presence of noise and variation.")

##################################################################################################################################
@st.cache(allow_output_mutation=True)
def load_model(id, fname):

    save_dest = Path('models')
    save_dest.mkdir(exist_ok=True)
    
    f_checkpoint = Path(f"models/{fname}")

    if not f_checkpoint.exists():
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            from GD_download import download_file_from_google_drive
            download_file_from_google_drive(id, f_checkpoint)

fname_ids = ['1wqinp1QqTsrwQGyN3HfjJXYt7AYw9PIv', 
            '1LuZVsB62C7QN6F8sDwiBWVfXVobY0Cnv',
            '1T2V-IOu8srfjwcdfbc58ENvPYXFU5Q_5',
            '1gBxiSgCo0z-8VEo7u-Ad1Je8j_tQCdWo',
            '1vFCr1WQ_u30ZM6SGn47-MIpn-CVebocg']

for fname_id in fname_ids:
    for i in range(5):
        load_model(fname_id, f'model_{i+1}.pth' )

# pseudo_probs = {}

# aptos2019_df = pd.read_csv('train.csv')
# aptos2019_img_paths = 'train/' + aptos2019_df['id_code'].values + '.png'
# aptos2019_labels = aptos2019_df['diagnosis'].values

# test_df = pd.read_csv('test.csv')
# test_img_paths = 'test/' + test_df['id_code'].values + '.png'
# test_labels = np.zeros(len(test_img_paths))

test_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

##########################################################################################
# Ensemble

def aptos_blind_detection(img):
    # load model
    model = get_model(model_name='se_resnext50_32x4d',
                    num_outputs=1,
                    pretrained=False,
                    freeze_bn=True,
                    dropout_p=0)
    model.eval()

    image_tensor = test_transform(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)

    probs = []
    for fold in range(5):
        model.load_state_dict(torch.load('models/model_%d.pth' % (fold+1), map_location=torch.device('cpu')))

        probs_fold = []
        with torch.no_grad():
            output = model(input)

            probs_fold.extend(output.data.cpu().numpy()[:, 0])
            probs_fold = np.array(probs_fold)
            probs.append(probs_fold)
    probs = np.mean(probs, axis=0)
    return probs

uploaded_file = st.file_uploader("Choose a retina image ...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded retina image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    probs = aptos_blind_detection(image)
    if probs < 0.5:
        st.write("The retina image has no diabetic retinopathy ")
    elif probs < 1.5:
        st.write("The retina image has mild diabetic retinopathy ")
    elif probs < 2.5:
        st.write("The retina image has moderate diabetic retinopathy ")    
    elif probs < 3.5:
        st.write("The retina image has severe diabetic retinopathy ")
    else:
        st.write("The retina image has proliferate diabetic retinopathy ")

