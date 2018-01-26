## Instance Segmentation ([Mask_RCNN](https://github.com/ascentai/Mask_RCNN))

Run the folowing to train the model on synthetic images.
It produces a weight file (`unreal_model_weights.h5`).

```python
python unreal_train.py
```

Run the following to generate 2D bounding box data (.npy) which is used by the 3D bounding box model.

```python
python unreal_bbox2d.py

