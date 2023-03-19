from imageai.Detection import ObjectDetection
import os
import csv

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()  
detector.setModelPath(os.path.join(execution_path,
                                   "resnet50_coco_best_v2.1.0.h5"))  
detector.loadModel()  
custom_objects = detector.CustomObjects(
    person=True)  

detections, extracted_images = detector.detectObjectsFromImage(
    input_image=os.path.join(execution_path, "Originals/poza1.jpg"),
    output_image_path=os.path.join(execution_path, "Afterimages/poza1crop.jpg"), minimum_percentage_probability=70,
    custom_objects=custom_objects, display_object_name=True, extract_detected_objects=True)
nume = "poza1.jpg"

with open('Saved_file.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Nume imagine", "Tip obiect", "Probabilitate", "Coordonate"]) 

for eachObject in detections:
    with open('Saved_file.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([nume, eachObject["name"], eachObject["percentage_probability"], eachObject[
            "box_points"]])  
    print("Tip obiect:", eachObject["name"], "; Probabilitate:", eachObject["percentage_probability"], "; Coordonate:",
          eachObject["box_points"]) 
