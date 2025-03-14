from fastapi import FastAPI, UploadFile, File
from trism import TritonModel
import numpy as np
from PIL import Image
from torchvision import transforms
import io

app = FastAPI()

# Create triton model.
model = TritonModel(
  model="densenet_trt",     # Model name.
  version=1,            # Model version.
  url="localhost:8001", # Triton Server URL.
  grpc=True             # Use gRPC or Http.
)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return np.expand_dims(preprocess(img).numpy(), axis=0)
  

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img_np = preprocess_image(contents)
    try:
        outputs = model.run(data=[img_np])
        inference_output = outputs['fc6_1'].astype(str)
        result = np.squeeze(inference_output)[:5]
        return {"prediction": result.tolist()}
    except Exception as e:
        return {"error": str(e)}