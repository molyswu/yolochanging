import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pickle
import copy
import yolo.config as cfg
import time
import tensorflow as tf
import threading

#Many producers and one consumer
class Pascal_voc(object):
    def __init__(self, phase, rebuild=False):
        self.devkil_path = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit')
        self.data_path = os.path.join(self.devkil_path, 'VOC2007')
        self.cache_path = cfg.CACHE_PATH
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.num_enqueue_threads = cfg.NUM_ENQUEUE_THREADS
        self.cell_size = cfg.CELL_SIZE
        self.classes = cfg.CLASSES
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))
        self.flipped = cfg.FLIPPED
        self.phase = phase
        self.rebuild = rebuild
        self.cursor = 0
        self.epoch = 1
        self.gt_labels = None
        self.gt_labels_length = 0
        self._lock = threading.Lock()
        self._next_idx = 0
        self.prepare()
	
    def get_one_image_label_element(self):
        current_index = 0
        with self._lock:
            current_index = self._next_idx
            self._next_idx += 1
               
        image = np.zeros(
            (self.image_size, self.image_size, 3))
        label = np.zeros(
            (self.cell_size, self.cell_size, 25))
        imname = self.gt_labels[current_index]['imname']
        flipped = self.gt_labels[current_index]['flipped']
        
        image[:, :, :] = self.image_read(imname, flipped)
        label[:, :, :] = self.gt_labels[current_index]['label']
        
        print("In one_image_label function , print out %d image and its label" % current_index)
        
        return (current_index, image, label)

    # def get_batch(self):
    #     # Define the operators
    #     # 这个单一的图片好像不是图内的一部分，这个有可能是错的
    #     (image, label) = self.get_one_image_label_element()
    #     # print('get one image and label from the file')
    #
    #     image_shape = (self.image_size, self.image_size, 3)  # possible value is a int number
    #
    #     label_size = (self.cell_size, self.cell_size, 25)  # possible value is 0 or 1
    #
    #     processed_queue = tf.FIFOQueue(capacity=int(self.batch_size * 1.5),
    #                                    shapes=[image_shape, label_size],
    #                                    dtypes=[tf.float32, tf.float32],
    #                                    name='processed_queue')
    #
    #
    #     enqueue_processed_op = processed_queue.enqueue([image, label])
    #     # print('enqueue one image and label to the FIFOQueue')
    #
    #     num_enqueue_threads = min(self.num_enqueue_threads, self.gt_labels_length)
    #
    #     queue_runner = tf.train.QueueRunner(processed_queue, [enqueue_processed_op] * num_enqueue_threads)
    #     tf.train.add_queue_runner(queue_runner)
    #
    #     (images, labels) = processed_queue.dequeue_many(self.batch_size)
    #
    #     labels = tf.Print(labels, data=[processed_queue.size()],
    #                       message="On machine %d function get_batch, Nb element left, input:" % self.task_index)
    #
    #     print('In function get_batch,dequeue_many batchsize')
    #     # 这个就是流程，只是这个输出应该没有用的阿。这个在一开始就经过了
    #     return (images, labels)


    def prepare(self):
        # timer_prepare = Timer()
        # timer_prepare.tic()
        start_prepare_time = time.time()
        gt_labels = self.load_labels()
        print()
        if self.flipped:
            print('Appending horizontally-flipped training examples ...')
            gt_labels_cp = copy.deepcopy(gt_labels)
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped'] = True
                gt_labels_cp[idx]['label'] = gt_labels_cp[idx]['label'][:, ::-1, :]
                for i in range(self.cell_size):
                    for j in range(self.cell_size):
                        if gt_labels_cp[idx]['label'][i, j, 0] == 1:
                            gt_labels_cp[idx]['label'][i, j, 1] = \
                                self.image_size - 1 - \
                                gt_labels_cp[idx]['label'][i, j, 1]
            gt_labels += gt_labels_cp
        np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        self.gt_labels_length = len(self.gt_labels)
        timer_prepare_elapse = time.time() - start_prepare_time
        print(
            "++++++++++++++++++++++++++++++++++++++++++++++++++++++Load labels(or read from cache) take %d s, together record %d Imagepath+Labelnotation lines +++++++++++++++++++++++" % (
                timer_prepare_elapse, len(gt_labels)))
        return gt_labels


# take 1 second.
# For training process, this take the trainval.txt, all lines in it, hereby all data set from it. we also append the imagenamepath as the first element. labels follow
    def load_labels(self):
        cache_file = os.path.join(self.cache_path, 'pascal_' + self.phase + '_gt_labels.pkl')
        
        if os.path.isfile(cache_file) and not self.rebuild:
            print('Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            return gt_labels
        
        print('Processing gt_labels from: ' + self.data_path)
        
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        
        if self.phase == 'train':
            txtname = os.path.join(
                self.data_path, 'ImageSets', 'Main', 'trainval.txt')
        else:
            txtname = os.path.join(
                self.data_path, 'ImageSets', 'Main', 'test.txt')
        with open(txtname, 'r') as f:
            self.image_index = [x.strip() for x in f.readlines()]
        
        gt_labels = []
        for index in self.image_index:
            label, num = self.load_pascal_annotation(index)
            if num == 0:
                continue
            imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
            gt_labels.append({'imname': imname,
                              'label': label,
                              'flipped': False})
        
        # Save to the file in the form of (key, values)
        print('Saving gt_labels to: ' + cache_file)
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_labels, f)
        return gt_labels

# work rarely, run once save in cache
    def load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        label dimension:  [cell_size,cell_size,numclasses+5]
        """
        imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        im = cv2.imread(imname)
        cv2.imshow('image', im)
        h_ratio = 1.0 * self.image_size / im.shape[0]
        w_ratio = 1.0 * self.image_size / im.shape[1]
        # im = cv2.resize(im, [self.image_size, self.image_size])
        
        label = np.zeros((self.cell_size, self.cell_size, 25))
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        
        for obj in objs:
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based。only one goal here that is make the 1 based pixel to 0 based and the same size as the original image
            x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, self.image_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, self.image_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, self.image_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, self.image_size - 1), 0)
            
            # which class it is, represented by number
            cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()]
            # cls_ind is the index of the 20 classes
            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
            # The (x,y) coordinates represent the center of the box relative to the bounds of the grid cell.So we have cell_size here. So it means the center x and center y will be in the range of cell size, relative to its real position in the image_size
            x_ind = int(boxes[0] * self.cell_size / self.image_size)
            y_ind = int(boxes[1] * self.cell_size / self.image_size)
            if label[y_ind, x_ind, 0] == 1:
                continue
            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1:5] = boxes
            # last one is the class index
            label[y_ind, x_ind, 5 + cls_ind] = 1
        
        return label, len(objs)


    def image_read(self, imname, flipped=False):
        '''
        better crops, scale, normalizes the given image.
        scale : The image wil be first scaled to this size.
            If isotropic is true, the smaller side is rescaled to this,
            preserving the aspect ratio.
        crop  : After scaling, a central crop of this size is taken.
        mean  : Subtracted from the image
        '''
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        '''
        offset = (new_shape - crop) / 2
        img = tf.slice(img, begin=tf.pack([offset[0], offset[1], 0]), size=tf.pack([crop, crop, -1]))
        # Mean subtraction, this is normalization
        return tf.to_float(img) - mean
        '''
        image = (image / 255.0) * 2.0 - 1.0
        if flipped:
            # means first and third dimension read as normal sequence, but second dimension read from right to left. So it is image flipped
            image = image[:, ::-1, :]
        return image
