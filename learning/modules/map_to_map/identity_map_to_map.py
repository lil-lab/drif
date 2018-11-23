from learning.modules.map_transformer_base import MapTransformerBase

sentence_embedding_size = 120
sentence_embedding_layers = 1
word_embedding_size = 20


class IdentityMapProcessor(MapTransformerBase):

    def __init__(self, source_map_size=32, world_size=32):
        super(IdentityMapProcessor, self).__init__(source_map_size, world_size)
        self.map_size = source_map_size
        self.world_size = world_size

    def init_weights(self):
        pass

    def cuda(self, device=None):
        MapTransformerBase.cuda(self, device)
        return self

    def forward(self, images, sentence_embeddimgs, map_poses, proc_mask=None, show=""):
        self.set_maps(images, map_poses)
        return images, map_poses