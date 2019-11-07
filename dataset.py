"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Author: Mike Danielczuk
"""

import os
import skimage
import numpy as np

"""
ImageDataset creates a Matterport dataset for a directory of
images in order to ensure compatibility with benchmarking tools 
and image resizing for networks.
Directory structure must be as follows:
$base_path/
    test_indices.npy
    train_indices.npy
    images/ (Train/Test Images here)
        image_000000.png
        image_000001.png
        ...
    dist_ims/ (GT dists here, one channel)
        image_000000.png
        image_000001.png
        ...
"""

class ImageDataset(object):
    
    def __init__(self, config):
        assert config['dataset']['path'] != "", "You must provide the path to a dataset!"

        self.dataset_config = config['dataset']
        self.base_path = config['dataset']['path']
        self.images = config['dataset']['images']
        self.dists = config['dataset']['dists']
        self._channels = config['model']['settings']['image_channel_count']

        self._image_ids = []
        self.image_info = []
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def prepare(self):
        """Prepares class for use."""

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })
    
    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def load(self, indices_file, augment=False):

        # Load the indices for imset.
        split_file = os.path.join(self.base_path, '{:s}'.format(indices_file))
        self.image_id = np.load(split_file)
        self.add_class('clutter', 1, 'fg')

        flips = [1, 2, 3]
        for i in self.image_id:
            if 'numpy' in self.images:
                p = os.path.join(self.base_path, self.images,
                                'image_{:06d}.npy'.format(i))
            else:
                p = os.path.join(self.base_path, self.images,
                                'image_{:06d}.png'.format(i))
            self.add_image('clutter', image_id=i, path=p)

            if augment:
                for flip in flips:
                    self.add_image('clutter', image_id=i, path=p, flip=flip)

    def flip(self, image, flip):
        # flips during training for augmentation

        if flip == 1:
            image = image[::-1,:,:]
        elif flip == 2:
            image = image[:,::-1,:]
        elif flip == 3:
            image = image[::-1,::-1,:]
        return image

    def load_image(self, image_id):
        # loads image from path
        if 'numpy' in self.images:
            image = np.load(self.image_info[image_id]['path']).squeeze()
        else:
            image = skimage.io.imread(self.image_info[image_id]['path'])
        
        if self._channels < 4 and image.shape[-1] == 4 and image.ndim == 3:
            image = image[...,:3]
        if self._channels == 1 and image.ndim == 2:
            image = image[:,:,np.newaxis]
        elif self._channels == 1 and image.ndim == 3:
            image = image[:,:,0,np.newaxis]
        elif self._channels == 3 and image.ndim == 3 and image.shape[-1] == 1:
            image = skimage.color.gray2rgb(image)
        elif self._channels == 4 and image.shape[-1] == 3:
            concat_image = np.concatenate([image, image[:,:,0:1]], axis=2)
            assert concat_image.shape == (image.shape[0], image.shape[1], image.shape[2] + 1), concat_image.shape
            image = concat_image
            
        return image

    def load_dist(self, image_id):
        # loads dist from path
        info = self.image_info[image_id]
        _image_id = info['id']
        Is = []
        file_name = os.path.join(self.base_path, self.dists,
          'image_{:06d}.png'.format(_image_id))

        dist = skimage.io.imread(file_name)
        return dist

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "clutter":
            return info["path"] + "-{:d}".format(info["flip"])
        else:
            return ""

    @property
    def indices(self):
        return self.image_id[:100]

    @property
    def image_ids(self):
        return self._image_ids[:100]
