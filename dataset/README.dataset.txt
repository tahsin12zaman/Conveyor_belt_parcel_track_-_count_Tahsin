# Boxes on a Conveyer Belt > Resize640_augmented10x
https://universe.roboflow.com/mohamed-traore-2ekkp/boxes-on-a-conveyer-belt

Provided by a Roboflow user
License: CC BY 4.0

## Use Cases:
This dataset can be used to track boxes on an assembly or manufacturing line, or as a starter-dataset for package detection and "defect" detection use cases for boxes.

* For the defect detection use case, one can Clone the images from this project to a new one, and add more examples (labels) of boxes with defects, such as: `damaged corners`, `unsealed boxes`, and more. This defect detection model can be built as a single object-detection model, or broken into a "two pass detection" model (identify the box and the defects with object detection --> send the cropped detections of the defects to a classification model to confirm the classification of the defect, and the severity)

* [Two Pass detection starter-code](https://github.com/roboflow-ai/roboflow-computer-vision-utilities/blob/main/images/twoPass.py)
* [Converting an Object Detection Dataset to Classification with Isolate Objects](https://blog.roboflow.com/isolate-objects/)
* [Roboflow: Single Label Classification](https://blog.roboflow.com/label-classification/) | [Roboflow: Multi-Label Classification](https://blog.roboflow.com/multi-label-classification/)
* [Roboflow: Dataset Types](https://help.roboflow.com/dataset-upload-roboflow-data-types) | [Adding Data: Classification](https://docs.roboflow.com/adding-data/classification)
* [Formats](https://roboflow.com/formats) | [Multi-Label Classification Format](https://roboflow.com/formats/multiclass-classification-csv) | [OpenAI CLIP Classification Format](https://roboflow.com/formats/openai-clip-classification)

### Classes:
* `box`