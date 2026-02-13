import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import heapq
from typing import NamedTuple
import torchvision.transforms as T
import os

from salad.eval import load_model # load salad

device = 'cuda'

tensor_transform = T.ToPILImage()
denormalize = T.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])

def input_transform(image_size=None):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    transform_list = [T.ToTensor(), T.Normalize(mean=MEAN, std=STD)]
    if image_size:
        transform_list.insert(0, T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR))
    return T.Compose(transform_list)

class LoopMatch(NamedTuple):
    similarity_score: float
    query_submap_id: int
    query_submap_frame: int
    detected_submap_id: int
    detected_submap_frame: int

class LoopMatchQueue:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.heap = []  # Simulated max-heap by negating scores

    def add(self, match: LoopMatch):
        # Negate similarity_score to turn min-heap into max-heap
        item = (-match.similarity_score, match)
        # item = (-match.detected_submap_id, match)
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, item)
        else:
            # Push new element and remove the largest (i.e., smallest negated)
            heapq.heappushpop(self.heap, item)

    def get_matches(self):
        """Return sorted list of matches (lowest value first)"""
        return [match for _, match in sorted(self.heap, reverse=True)]
        

class ImageRetrieval:
    def __init__(self, input_size=224):

        ckpt_pth = os.path.join(torch.hub.get_dir(), "checkpoints/dino_salad.ckpt")
        self.model = load_model(ckpt_pth)
        self.model.eval()
        self.transform = input_transform((input_size, input_size))

    def get_single_embeding(self, cv_img):
        with torch.no_grad():
            pil_img = self.transform(tensor_transform(cv_img))
            return self.model(pil_img.to(device))

    def get_batch_descriptors(self, imgs):
        # Expecting imgs to be a batch of images (B, C, H, W)
        with torch.no_grad():
            pil_imgs = [tensor_transform(img) for img in imgs]  # Convert each tensor to PIL Image
            imgs = torch.stack([self.transform(img) for img in pil_imgs])  # Apply transform and stack
            return self.model(imgs.to(device))
    
    def get_all_submap_embeddings(self, submap):
        # Frames is np array of shape (S, 3, H, W)
        frames = submap.get_all_frames()
        return self.get_batch_descriptors(frames)

    def find_loop_closures(self, map, submap, max_similarity_thres = 0.80, max_loop_closures = 0):
        matches_queue = LoopMatchQueue(max_size=max_loop_closures)
        query_id = 0
        for query_vector in submap.get_all_retrieval_vectors():
            best_score, best_submap_id, best_frame_id = map.retrieve_best_score_frame(query_vector, submap.get_id(), ignore_last_submap=True)
            if best_score < max_similarity_thres:
                new_match_data = LoopMatch(best_score, submap.get_id(), query_id, best_submap_id, best_frame_id)
                matches_queue.add(new_match_data)
            query_id += 1
        
        return matches_queue.get_matches()