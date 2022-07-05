import torch
import os
import imageio
import numpy as np
from torchvision import transforms

from Architecture import TransformerModel
from Processing import process_image, indices_to_text

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHANNELS = 1
HIDDEN = 512
ENC_LAYERS = 2
DEC_LAYERS = 2
N_HEADS = 4
LENGTH = 32
ALPHABET = ['PAD', 'SOS', ' ', '!', '"', '%', '(', ')', ',', '-', '.', '/',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?',
            '[', ']', '«', '»', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И',
            'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х',
            'Ц', 'Ч', 'Ш', 'Щ', 'Э', 'Ю', 'Я', 'а', 'б', 'в', 'г', 'д', 'е',
            'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т',
            'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я',
            'ё', 'EOS']
MODEL_PATH = "model/model_ocr_rus_handwriting.pt"
idx2char = {idx: char for idx, char in enumerate(ALPHABET)}
model = TransformerModel(len(ALPHABET), hidden=HIDDEN, enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS,
                         nhead=N_HEADS, dropout=0.0).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))


def prediction(folder_images):
    preds = {}
    model.eval()

    with torch.no_grad():
        for filename in os.listdir(folder_images):

            img = imageio.v2.imread(folder_images + filename, as_gray=False, pilmode="RGB")

            img = process_image(np.asarray(img)).astype('uint8')
            img = img / img.max()
            img = np.transpose(img, (2, 0, 1))

            src = torch.FloatTensor(img).unsqueeze(0).to(DEVICE)
            if CHANNELS == 1:
                src = transforms.Grayscale(CHANNELS)(src)
            out_indexes = model.predict(src)
            pred = indices_to_text(out_indexes[0], idx2char)
            preds[filename] = pred

    return preds
