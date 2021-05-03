import torch
import torch.nn as nn
from torchvision.models import resnet18
from tars.base.model import Model
from tars.auxilary_models.mask_rcnn import MaskRCNN
from tars.auxilary_models import SegmentationModel
from tars.auxilary_models.embed_and_readout import EmbedAndReadout


class VisionModule(Model):
    def __init__(self, num_objects, object_na_idx, conf):
        super(VisionModule, self).__init__()
        self.max_img_objects = conf.max_img_objects
        self.raw_vision_features_size = conf.raw_vision_features_size
        self.object_na_idx = object_na_idx

        self.vision_cnn = resnet18(pretrained=True)
        assert self.vision_cnn.fc.in_features == self.raw_vision_features_size
        self.vision_cnn.fc = nn.Sequential()

        self.object_embed_and_readout = EmbedAndReadout(
            dict_size=num_objects,
            embed_dim=conf.object_emb_dim,
            out_dim=conf.vision_object_emb_dim,
            padding_idx=object_na_idx,
            max_len=conf.max_img_objects,
            conf=conf,
            use_pe=False
        )

        self.use_instance_seg = conf.use_instance_seg
        if self.use_instance_seg:
            self.detection_model = MaskRCNN(model_load_path=conf.detection_model_path)
        else:
            self.detection_model = SegmentationModel(
                                    model_load_path=conf.detection_model_path
                                )
        self.detection_model.eval()

        self.vision_mixer = nn.Linear(
                            conf.raw_vision_features_size + conf.vision_object_emb_dim,
                            conf.vision_features_size
                        )

    def forward(self, img):
        # can run this on all images and cache on disk to save training/eval time

        raw_vision_features = self.vision_cnn(img)
        assert raw_vision_features.shape == \
            torch.Size([img.shape[0], self.raw_vision_features_size])

        with torch.no_grad():
            if self.use_instance_seg:
                raise NotImplementedError
            else:
                seg_img = self.detection_model(img)
                seg_img = seg_img.argmax(1)
                objects = []

                for i in range(seg_img.shape[0]):  # FIXME: can we avoid loop?
                    img_objects, counts = seg_img[i].unique(return_counts=True)
                    # pick self.max_img_objects largest objects
                    img_objects = img_objects[counts.sort(descending=True)[1]][:self.max_img_objects]
                    padding = torch.tensor(
                                [self.object_na_idx] * \
                                    (self.max_img_objects - len(img_objects))
                            )

                    objects.append(torch.cat((img_objects, padding)))

                objects = torch.stack(objects)

        objects_readout = self.object_embed_and_readout.forward(objects.int())

        out = self.vision_mixer(
                torch.cat((objects_readout, raw_vision_features), dim=1)
            )

        return out

    def get_img_transforms(self):
        return self.detection_model.get_img_transforms()
