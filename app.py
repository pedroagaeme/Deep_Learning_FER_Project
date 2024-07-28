from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
import os
import torch
from torch import nn
from torchvision.models import vgg19, VGG19_Weights, resnet18, ResNet18_Weights, efficientnet_b0, EfficientNet_B0_Weights, vit_b_16, ViT_B_16_Weights
from torchvision.transforms import v2
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'dj30mdsmdxda'
NUM_CLASSES = 7
IMAGE_SIZE = 224

number_to_class = ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral']
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

preprocess = v2.Compose([
    v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean, std)
])

class VGG(nn.Module):
    def __init__(self, num_classes):
        super(VGG, self).__init__()
        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT)
        self.vgg.classifier[-1] = nn.Linear(4096, num_classes, bias=True)

    def forward(self, x):
        return self.vgg(x)

vgg_model = VGG(NUM_CLASSES)
vgg_checkpoint = torch.load('models/VGG19_best_model.pth', map_location=torch.device('cpu'))
vgg_model.load_state_dict(vgg_checkpoint)
vgg_model.eval()

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.resnet(x)

resNet_model = ResNet18(NUM_CLASSES)
resnet_checkpoint = torch.load('models/ResNet18_best_model.pth', map_location=torch.device('cpu'))
resNet_model.load_state_dict(resnet_checkpoint)
resNet_model.eval()

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB0, self).__init__()
        self.efficientNet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.efficientNet.classifier[1] = nn.Linear(self.efficientNet.classifier[1].in_features, num_classes, bias=True)
    def forward(self, x):
        return self.efficientNet(x)

efficientNet_model = EfficientNetB0(NUM_CLASSES)
efficientNet_checkpoint = torch.load('models/EfficientNet_B0_best_model.pth', map_location=torch.device('cpu'))
efficientNet_model.load_state_dict(efficientNet_checkpoint)
efficientNet_model.eval()

class ViT(nn.Module):
  def __init__(self, num_classes):
    super(ViT, self).__init__()
    self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    self.vit.heads[0] = nn.Linear(self.vit.heads[0].in_features, num_classes)

  def forward(self, x):
    return self.vit(x)

vit_model = ViT(NUM_CLASSES)
viT_checkpoint = torch.load('models/ViT_best_model.pth', map_location=torch.device('cpu'))
vit_model.load_state_dict(viT_checkpoint)
vit_model.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_and_preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor

@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('prediction', filename=filename))
        else:
            flash('Invalid file extension')
            return redirect(request.url)
    return render_template('index.html')

@app.route('/prediction/<filename>')
def prediction(filename):
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img_tensor = load_and_preprocess_image(img_path)

    vgg_probabilities = []
    resNet_probabilities = []
    efficientNet_probabilities = []
    vit_probabilities = []

    with torch.no_grad():
        vgg_outputs = vgg_model(img_tensor)
        resNet_outputs = resNet_model(img_tensor)
        efficientNet_outputs = efficientNet_model(img_tensor)
        vit_outputs = vit_model(img_tensor)

        vgg_probabilities.append(torch.nn.functional.softmax(vgg_outputs[0], dim=0))
        resNet_probabilities.append(torch.nn.functional.softmax(resNet_outputs[0], dim=0))
        efficientNet_probabilities.append(torch.nn.functional.softmax(efficientNet_outputs[0], dim=0))
        vit_probabilities.append(torch.nn.functional.softmax(vit_outputs[0], dim=0))

    vgg_probabilities = torch.mean(torch.stack(vgg_probabilities), dim=0)
    resNet_probabilities = torch.mean(torch.stack(resNet_probabilities), dim=0)
    efficientNet_probabilities = torch.mean(torch.stack(efficientNet_probabilities), dim=0)
    vit_probabilities = torch.mean(torch.stack(vit_probabilities), dim=0)

    vgg_prob_values, vgg_class_indices = vgg_probabilities.sort(descending=True)
    resNet_prob_values, resNet_class_indices = resNet_probabilities.sort(descending=True)
    efficientNet_prob_values, efficientNet_class_indices = efficientNet_probabilities.sort(descending=True)
    vit_prob_values, vit_class_indices = vit_probabilities.sort(descending=True)

    predictions = {
        "vgg_class1": number_to_class[vgg_class_indices[0]],
        "vgg_class2": number_to_class[vgg_class_indices[1]],
        "vgg_class3": number_to_class[vgg_class_indices[2]],
        "vgg_prob1": "{:.8f}".format(vgg_prob_values[0].item()),
        "vgg_prob2": "{:.8f}".format(vgg_prob_values[1].item()),
        "vgg_prob3": "{:.8f}".format(vgg_prob_values[2].item()),
        "resNet_class1": number_to_class[resNet_class_indices[0]],
        "resNet_class2": number_to_class[resNet_class_indices[1]],
        "resNet_class3": number_to_class[resNet_class_indices[2]],
        "resNet_prob1": "{:.8f}".format(resNet_prob_values[0].item()),
        "resNet_prob2": "{:.8f}".format(resNet_prob_values[1].item()),
        "resNet_prob3": "{:.8f}".format(resNet_prob_values[2].item()),
        "efficientNet_class1": number_to_class[efficientNet_class_indices[0]],
        "efficientNet_class2": number_to_class[efficientNet_class_indices[1]],
        "efficientNet_class3": number_to_class[efficientNet_class_indices[2]],
        "efficientNet_prob1": "{:.8f}".format(efficientNet_prob_values[0].item()),
        "efficientNet_prob2": "{:.8f}".format(efficientNet_prob_values[1].item()),
        "efficientNet_prob3": "{:.8f}".format(efficientNet_prob_values[2].item()),
        "vit_class1": number_to_class[vit_class_indices[0]],
        "vit_class2": number_to_class[vit_class_indices[1]],
        "vit_class3": number_to_class[vit_class_indices[2]],
        "vit_prob1": "{:.8f}".format(vit_prob_values[0].item()),
        "vit_prob2": "{:.8f}".format(vit_prob_values[1].item()),
        "vit_prob3": "{:.8f}".format(vit_prob_values[2].item())
    }

    return render_template('predict.html', predictions=predictions)

if __name__ == "__main__":
    app.run(debug=True)
