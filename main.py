from source.model import DepthMap
from source.constants import IMG_WIDTH, IMG_HEIGHT

import streamlit as st
import torch
import torchvision.transforms as tfs
from torchvision.utils import save_image
from skimage import io
from skimage.transform import resize


@st.cache_resource
def get_model():
    model = DepthMap()
    model.load_state_dict(torch.load('weights/Linknet(resnet18).pt'))
    model.eval()
    return model


left, right = st.columns(2)

with left:
    left_image = st.file_uploader(label='Левое изображение', )

    if left_image:
        st.image(left_image)
        with open('images/left.png', 'wb') as f:
            f.write(left_image.read())

with right:
    right_image = st.file_uploader(label='Правое изображение', )

    if right_image:
        st.image(right_image)
        with open('images/right.png', 'wb') as f:
            f.write(right_image.read())

button = st.button(label='Построить карту смещений', use_container_width=True)
if button:
    transforms = tfs.Compose([
        tfs.ToPILImage(),
        tfs.Resize(IMG_WIDTH),
        tfs.CenterCrop((IMG_HEIGHT, IMG_WIDTH)),
        tfs.ToTensor(),
    ])
    left_image = io.imread('images/left.png')
    left_image = transforms(left_image)
    left_image = left_image.unsqueeze(0)

    right_image = io.imread('images/right.png')
    right_image = transforms(right_image)
    right_image = right_image.unsqueeze(0)

    predict = get_model()(left_image, right_image)
    predict = predict.squeeze(0).permute(1, 2, 0).detach().numpy()
    io.imsave(fname='predicted.jpg', arr=predict)
    predict = io.imread(fname='predicted.jpg')
    predict = resize(predict, (441, 800))
    st.image(predict, clamp=True, use_column_width=True)
