import numpy as np
import logging
import os
from data_downloader import download_data

def load_img(img_dir, img_list):
    images=[]
    for i, image_name in enumerate(img_list):
        image = np.load(img_dir+image_name)
        images.append(image)
        
    images = np.array(images)
    
    return(images)

def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
    L = len(img_list)

    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
                       
            X = load_img(img_dir, img_list[batch_start:limit]).astype(np.float32)
            Y = load_img(mask_dir, mask_list[batch_start:limit]).astype(np.float32)

            yield (X,Y)

            batch_start += batch_size   
            batch_end += batch_size
            
def load_data(args):
    train_data_num = 0
    test_data_num = 0
    train_data_global = None
    test_data_global = None
    train_data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    nc = args.output_dim

    if args.process_id == 0:  # server
        logging.info("load data for server test")
        download_data("test_3D.rar" , os.path.join(".", ""))
        data_dir = "."
        data_dir_test = os.path.join(data_dir, 'test_3D')
        data_dir_test_image = os.path.join(data_dir_test, 'images', "")
        data_dir_test_mask = os.path.join(data_dir_test, 'masks', "")
        BATCH_SIZE_TEST = args.batch_size
        
        test_img_list= sorted(os.listdir(data_dir_test_image))
        test_mask_list = sorted(os.listdir(data_dir_test_mask))
        
        test_generator = imageLoader(data_dir_test_image, test_img_list, 
                                data_dir_test_mask, test_mask_list, BATCH_SIZE_TEST)
    
        logging.error("server - test_generator created")
        
        client_idx = int(args.process_id) - 1
        test_data_num = len(test_img_list)
        test_data_global = test_generator
        test_data_local_dict[client_idx] =test_generator
        return (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            nc,
        )
    else:  # client
        client_idx = int(args.process_id) - 1
        logging.info(f"load center {int(client_idx)} data")
        if client_idx==0:
            download_data("client1_3D.rar", os.path.join(".", ""))
            args.data_cache_dir = os.path.join(".", 'client1_3D')
        elif client_idx==1:
            download_data("client2_3D.rar", os.path.join(".", ""))
            args.data_cache_dir = os.path.join(".", 'client2_3D')
        elif client_idx==2:
            download_data("client3_3D.rar", os.path.join(".", ""))
            args.data_cache_dir = os.path.join(".", 'client3_3D')
        data_dir = args.data_cache_dir
        data_dir_train = os.path.join(data_dir, 'train')
        data_dir_train_image = os.path.join(data_dir_train, 'imagesTr', "")
        data_dir_train_mask = os.path.join(data_dir_train, 'labelsTr', "")
        
        data_dir_val = os.path.join(data_dir, 'val')
        data_dir_val_image = os.path.join(data_dir_val, 'imagesTr', "")
        data_dir_val_mask = os.path.join(data_dir_val, 'labelsTr', "")
    
        BATCH_SIZE_TRAIN = args.batch_size
        BATCH_SIZE_TEST = args.batch_size
    
        train_img_list= sorted(os.listdir(data_dir_train_image))
        train_mask_list = sorted(os.listdir(data_dir_train_mask))
    
        train_generator = imageLoader(data_dir_train_image, train_img_list, 
                                        data_dir_train_mask, train_mask_list, BATCH_SIZE_TRAIN)
        
        logging.error("client - train_generator created")
        
        val_img_list= sorted(os.listdir(data_dir_val_image))
        val_mask_list = sorted(os.listdir(data_dir_val_mask))
        
        val_generator = imageLoader(data_dir_val_image, val_img_list, 
                                    data_dir_val_mask, val_mask_list, BATCH_SIZE_TRAIN)
        
        # test_img_list= sorted(os.listdir(data_dir_test_image))
        # test_mask_list = sorted(os.listdir(data_dir_test_mask))
        
        # test_generator = imageLoader(data_dir_test_image, test_img_list, 
        #                             data_dir_test_mask, test_mask_list, BATCH_SIZE_TEST)
        
        logging.error("client - test_generator created")
        
        train_data_num = len(train_img_list)
        # val_data_num = len(val_img_list)
        test_data_num = len(val_img_list)
        train_data_global = train_generator
        # val_data_global = val_generator
        test_data_global = val_generator
        train_data_local_num_dict[client_idx] = train_data_num
        # val_data_local_dict[client_idx] = val_generator
        train_data_local_dict[client_idx] = train_generator
        test_data_local_dict[client_idx] = val_generator
        logging.error(f"train_data_num: {train_data_num}")
        logging.error(f"test_data_num: {test_data_num}")
        logging.error(f"train_data_global: {train_data_global}")
        logging.error(f"test_data_global: {test_data_global}")
        logging.error(f"data_local_num_dict: {train_data_local_num_dict}")
        logging.error(f"train_data_local_dict: {train_data_local_dict}")
        logging.error(f"test_data_local_dict: {test_data_local_dict}")
        logging.error(f"nc: {nc}")

    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        nc,
    )