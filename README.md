This is mostly the engineering work to combine several state of art deep learning tasks together, and provide an surveillance system with given video input and sample person images.
In details:
- person detection code modified from https://github.com/tensorflow/models/tree/master/research/object_detection
- face detection code modified from https://github.com/Joker316701882/Additive-Margin-Softmax
- person reid code modified from https://github.com/VisualComputingInstitute/triplet-reid


And here is a quick tutorial of usage

i). `python3 --video path/to/video --frame_dir data/frames --person_dir data/persons --yaml_dir data/yamls`
- --video: input video under surveillance
- --frame_dir: decoded per frame jpg images from input video
- --person_dir: decoded per person image from each frame image, resized to 256x128
- --yaml_dir: per frame yaml file, to store information like image path, per person embedding

ii). pick person of interest, and store his/hers sample images into a yaml file with format like
```
name1:
    - path_to_image_for_name1
    - path_to_another_image_for_name1
name2:
    - path_to_image_for_name2
```

iii). `python3 main.py --yaml_dir data/yamls --namelist path/to/person_yaml`
- --yaml_dir: load information from previous generated yamls, update them with most likely person of interest and the distance
- --namelist: person of interest yaml file generated in step 2
- results frames with bounding box stored under data/output/frames

iv). `python3 main.py --yaml_dir data/yamls --distance 9.0 --video_out`
- --yaml_dir: load information from previous generated yamls
- --distance: manually chosen distance to best spot person of interest with min error
- --video_out: combine each frame with bounding box into data/output/video.mp4
