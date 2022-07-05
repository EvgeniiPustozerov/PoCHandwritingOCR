import glob
import os
import random
import shutil
from pathlib import Path

import streamlit as st
from Model import prediction

if st.button('Try random samples from the database'):
    folder = "data/sample/"
    list_all_audio = glob.glob("data/dataset/*.png")
    chosen_files = sorted(random.sample(list_all_audio, 3))
    for f in glob.glob(folder + '*'):
        os.remove(f)
    for f in chosen_files:
        shutil.copy2(f, folder)
        st.image(f)
    preds = prediction(folder)
    print(preds)
    st.write(preds)
uploaded_file = st.file_uploader("Choose your image with Russian text",
                                 accept_multiple_files=False, type=["png", "jpeg", "jpg"])
if uploaded_file is not None:
    folder = "data/user_data/"
    for f in glob.glob(folder + '*'):
        os.remove(f)
    bytes_data = uploaded_file.read()
    st.image(bytes_data)
    save_path = Path(folder, uploaded_file.name)
    with open(save_path, mode='wb') as w:
        w.write(uploaded_file.getvalue())
    preds = prediction(folder)
    print(preds)
    st.write(preds)
