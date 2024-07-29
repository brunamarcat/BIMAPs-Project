import flammkuchen as fl
import albumentations as A
import copy
import numpy as np

from deepd3.training.stream import DataGeneratorStream


class DataGeneratorStreamSiamese(DataGeneratorStream):
    def __getitem__(self, index):
        """Generate one batch of data
        
        Parameters
        ----------
        index : int
            batch index in image/label id list
        
        Returns
        -------
        tuple
            Contains two numpy arrays,
            each of shape (batch_size, height, width, 1). 
        """
        X = []
        Y0 = []
        Y1 = []
        eps = 1e-5

        if self.shuffle is False:
            np.random.seed(index)

        # Create all pairs in a given batch
        for i in range(self.batch_size):
            # Retrieve a single sample pair
            anchor_image, anchor_dendrite, anchor_spines = self.getSample()
            positive_image, positive_dendrite, positive_spines= copy.deepcopy(anchor_image), copy.deepcopy(anchor_dendrite), copy.deepcopy(anchor_spines)
            
            # TO DO: assegurar que la imatge es diferent
            negative_image, negative_dendrite, negative_spines = self.getSample()
#            negative_image, negative_dendrite, negative_spines = copy.deepcopy(anchor_image), copy.deepcopy(anchor_dendrite), copy.deepcopy(anchor_spines)  

            # # Augmenting the data
            # augmented = self.aug(image=positive_image.astype(np.uint8), 
            #     mask1=positive_dendrite.astype(np.uint8), 
            #     mask2=positive_spines.astype(np.uint8)) #augment positive
            
            # positive_image = augmented['image']
            # positive_dendrite = augmented['mask1']
            # positive_spines = augmented['mask2']

            if self.augment:
                augmented = self.aug(image=anchor_image.astype(np.uint8), 
                    mask1=anchor_dendrite.astype(np.uint8), 
                    mask2=anchor_spines.astype(np.uint8)) #augment anchor
                
                # anchor_image = augmented['image']
                # anchor_dendrite = augmented['mask1']
                # anchor_spines = augmented['mask2']
                
                # augmented = self.aug(image=positive_image.astype(np.uint8), 
                #     mask1=positive_dendrite.astype(np.uint8), 
                #     mask2=positive_spines.astype(np.uint8)) #augment positive
            
                # positive_image = augmented['image']
                # positive_dendrite = augmented['mask1']
                # positive_spines = augmented['mask2']
                
                augmented = self.aug(image=negative_image.astype(np.uint8), 
                    mask1=negative_dendrite.astype(np.uint8), 
                    mask2=negative_spines.astype(np.uint8)) #augment negative
                
                negative_image = augmented['image']
                negative_dendrite = augmented['mask1']
                negative_spines = augmented['mask2']

            # Min/max scaling
            anchor_image = (anchor_image.astype(np.float32) - anchor_image.min()) / (anchor_image.max() - anchor_image.min() + eps)
            positive_image = (positive_image.astype(np.float32) - positive_image.min()) / (positive_image.max() - positive_image.min() + eps)
            negative_image = (negative_image.astype(np.float32) - negative_image.min()) / (negative_image.max() - negative_image.min() + eps)
            
            # Shifting and scaling
            anchor_image = anchor_image * (self.normalize[1]-self.normalize[0]) + self.normalize[0]
            positive_image = positive_image * (self.normalize[1]-self.normalize[0]) + self.normalize[0]
            negative_image = negative_image * (self.normalize[1]-self.normalize[0]) + self.normalize[0]
            
            X.append(anchor_image)
            X.append(positive_image)
            X.append(negative_image)
            Y0.append(anchor_dendrite.astype(np.float32) / (anchor_dendrite.max() + eps))
            Y0.append(positive_dendrite.astype(np.float32) / (positive_dendrite.max() + eps))
            Y0.append(negative_dendrite.astype(np.float32) / (negative_dendrite.max() + eps))
            Y1.append(anchor_spines.astype(np.float32) / (anchor_spines.max() + eps)) # to ensure binary targets
            Y1.append(positive_spines.astype(np.float32) / (positive_spines.max() + eps))
            Y1.append(negative_spines.astype(np.float32) / (negative_spines.max() + eps))
            
        return np.asarray(X, dtype=np.float32)[..., None], (np.asarray(Y0, dtype=np.float32)[..., None],
                np.asarray(Y1, dtype=np.float32)[..., None])


    def _get_augmenter(self):
        """Defines used augmentations"""
        aug = A.Compose([
            A.RandomBrightnessContrast(p=0.25),    
            A.Blur(p=0.2),
            A.GaussNoise(p=0.5)], p=1,
            additional_targets={
                'mask1': 'mask',
                'mask2': 'mask'
            })
        return aug
