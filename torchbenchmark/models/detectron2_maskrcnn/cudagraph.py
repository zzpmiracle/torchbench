import torch

CUDA_GRAPH_CONFIG = {
    CUDA_GRAPH_NUM_SHAPES_PER_ORIENTATION: 0,
}

class GraphedWrapper(torch.nn.Module):
    def __init__(self, model_segment, graphed_forwards):
        super().__init__()
        self.model_segment = model_segment
        self.graphed_forwards = graphed_forwards

    def forward(self, images_tensor, image_sizes_tensor):
        shape = tuple(list(images_tensor.shape))
        if shape in self.graphed_forwards:
            return self.graphed_forwards[shape](images_tensor, image_sizes_tensor)
        elif images_tensor.shape[0] < images_per_gpu:
            # run with padding in case of in-complete batch
            # pad
            before_pad = images_tensor.shape[0]
            images_tensor = torch.nn.functional.pad(images_tensor, (0,0,0,0,0,0,0,images_per_gpu-before_pad))
            image_sizes_tensor = torch.nn.functional.pad(image_sizes_tensor, (0,0,0,images_per_gpu-before_pad))
            # run with graph
            shape = tuple(list(images_tensor.shape))
            if shape in self.graphed_forwards:
                out = self.graphed_forwards[shape](images_tensor, image_sizes_tensor)
            else:
                out = self.model_segment.eager_forward(images_tensor, image_sizes_tensor)
            # unpad
            out = [o[0:before_pad] for o in out]
            return out
        else:
            return self.model_segment.eager_forward(images_tensor, image_sizes_tensor)

def get_graphed_forwards():
    pass

def get_graphable_model(model, graphed_forwards):
    return GraphedWrapper(model, graphed_forwards)