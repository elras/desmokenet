"""
Desmokenet demo file

For more information see: S. Bolkar, C. Wang, F. A. Cheikh and S. Yildirim, 
                          Deep Smoke Removal from Minimally Invasive Surgery Videos,
                          2018 25th IEEE International Conference on Image Processing (ICIP), 
                          Athens, Greece, 2018, pp. 3403 3407. doi: 10.1109/ICIP.2018.8451815
"""
import os
import numpy as np
import caffe
from PIL import Image

input_path = "data/input/"
output_path = "data/result/"
model_path = "model/"


def EditFcnProto(templateFile, height, width):
    with open(templateFile, "r") as ft:
        template = ft.read()
    outFile = os.path.join(model_path, "DeployT.prototxt")
    with open(outFile, "w") as fd:
        fd.write(template.format(height=height, width=width))


if __name__ == "__main__":
    caffe.set_mode_cpu()

    image_folder = os.listdir(input_path)
    for imagename in image_folder:

        image = caffe.io.load_image(os.path.join(input_path, imagename))
        height = image.shape[0]
        width = image.shape[1]

        # Adapt the model according to input size
        EditFcnProto(os.path.join(model_path, "test_template.prototxt"), height, width)

        net = caffe.Net(
            os.path.join(model_path, "DeployT.prototxt"),
            os.path.join(model_path, "model_iter_40000.caffemodel"),
            caffe.TEST,
        )
        batchdata = []
        batchdata.append(image.transpose((2, 0, 1)))
        net.blobs["data"].data[...] = batchdata

        # Inference
        net.forward()

        result_float = net.blobs["sum"].data[0].transpose((1, 2, 0))
        result = Image.fromarray((result_float * 255.0).astype(dtype="uint8"))
        result.save(os.path.join(output_path, imagename))

