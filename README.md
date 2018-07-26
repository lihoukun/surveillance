This is mostly the engineering work to combine several state of art deep learning tasks together, and provide an surveillance system with given video input and sample person images.
In details:
- person detection code modified from https://github.com/tensorflow/models/tree/master/research/object_detection
- face detection code modified from https://github.com/Joker316701882/Additive-Margin-Softmax
- person reid code modified from https://github.com/VisualComputingInstitute/triplet-reid


And here is a quick tutorial of usage

i). `python3 main.py --video path/to/video --dump_person --debug`
- --video: input video under surveillance
- --dump_person: per person image dump 
- --debug: debug mode, in which intermidtae results will get dumped

ii). pick person of interest, and store his/hers sample images into a directory where each sub folder is the person's name, and inside each sub folder are images of that person (256x128)
```
trace_dir
    - person1_dir
        - image1
        - image2
    - person2_dir
        - image1
```

iii). `python3 main.py --name_dir path/to/trace_dir --dump_video --debug`
- --name_dir: trace dir as is in step 2
- --dump_video: save each frame with bounding box (shown person name), and combine all frames into a video
