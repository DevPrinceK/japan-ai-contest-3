# app.py
from flask import Flask, render_template, request, send_from_directory
import torch
from torchvision.utils import save_image
import os
from generator import Generator

app = Flask(__name__)
OUTPUT_DIR = "static/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Generator().to(device)
model.load_state_dict(torch.load("generator.pth", map_location=device))
model.eval()

# Route: Home Page
@app.route("/", methods=["GET", "POST"])
def index():
    images = []
    if request.method == "POST":
        digit = request.form.get("digit")
        if digit.isdigit() and 0 <= int(digit) <= 9:
            # Generate 5 images
            z = torch.randn(5, 100).to(device)
            with torch.no_grad():
                fake_imgs = model(z).view(-1, 1, 28, 28)
                fake_imgs = (fake_imgs + 1) / 2  # Rescale to [0,1]

                # Save each image
                for i, img in enumerate(fake_imgs):
                    path = f"{OUTPUT_DIR}/{digit}_{i}.png"
                    save_image(img, path)
                    images.append(path)

    return render_template("index.html", images=images)

# Serve images
@app.route('/static/images/<filename>')
def get_image(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True)
