"""
create train or eval dataset.
"""
import os
import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2

def create_dataset(dataset_path, config, training=True, buffer_size=1000):
    """
    create a train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        config(struct): the config of train and eval in diffirent platform.

    Returns:
        train_dataset, val_dataset
    """
    data_path = os.path.join(dataset_path, 'train' if training else 'test')
    ds = de.ImageFolderDataset(data_path, num_parallel_workers=4, class_indexing=config.class_index)
    resize_height = config.image_height
    resize_width = config.image_width
    
    # define operations mapping to each sample
    normalize_op = C.Normalize(mean=[0.485*255, 0.456*255, 0.406*255], std=[0.229*255, 0.224*255, 0.225*255])
    change_swap_op = C.HWC2CHW()
    type_cast_op = C2.TypeCast(mstype.int32)

    if training:
        # operations for training
        crop_decode_resize = C.RandomCropDecodeResize(resize_height, scale=(0.08, 1.0), ratio=(0.75, 1.333))
        horizontal_flip_op = C.RandomHorizontalFlip(prob=0.5)
        color_adjust = C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4)
    
        train_trans = [crop_decode_resize, horizontal_flip_op, color_adjust, normalize_op, change_swap_op]
        train_ds = ds.map(input_columns="image", operations=train_trans, num_parallel_workers=8)
        train_ds = train_ds.map(input_columns="label", operations=type_cast_op, num_parallel_workers=8)
        
        # apply shuffle operations
        train_ds = train_ds.shuffle(buffer_size=buffer_size)
        # apply batch operations
        ds = train_ds.batch(config.batch_size, drop_remainder=True)
    else:
        # operations for inference
        decode_op = C.Decode()
        resize_op = C.Resize((int(resize_width/0.875), int(resize_width/0.875)))
        center_crop = C.CenterCrop(resize_width)
        
        eval_trans = [decode_op, resize_op, center_crop, normalize_op, change_swap_op]
        eval_ds = ds.map(input_columns="image", operations=eval_trans, num_parallel_workers=8)
        eval_ds = eval_ds.map(input_columns="label", operations=type_cast_op, num_parallel_workers=8)
        ds = eval_ds.batch(config.eval_batch_size, drop_remainder=True)

    return ds
