import torch
import torch.nn as nn
from torchvision.models import resnet18
from tars.base.model import Model
from tars.auxilary_models import MultiLabelClassifier
from tars.auxilary_models.embed_and_readout import EmbedAndReadout


class VisionModule(Model):
    def __init__(self, num_objects, object_na_idx, conf):
        super(VisionModule, self).__init__()
        self.max_img_objects = conf.max_img_objects
        self.raw_vision_features_size = conf.raw_vision_features_size
        self.object_na_idx = object_na_idx
        self.remove_vision_readout = conf.remove_vision_readout

        self.vision_cnn = resnet18(pretrained=True)
        for n, p in self.vision_cnn.named_parameters():
            if n.startswith('layer1') or n.startswith('layer2'):
                p.requires_grad = False

        assert self.vision_cnn.fc.in_features == self.raw_vision_features_size
        self.vision_cnn.fc = nn.Sequential()

        if not self.remove_vision_readout:
            self.object_embed_and_readout = EmbedAndReadout(
                dict_size=num_objects,
                embed_dim=conf.object_emb_dim,
                out_dim=conf.vision_object_emb_dim,
                padding_idx=object_na_idx,
                history_max_len=conf.max_img_objects,
                dropout=conf.vision_readout_dropout,
                policy_conf=conf,
                use_pe=False
            )

            self.detection_model = MultiLabelClassifier(model_load_path=conf.detection_model_path)
            self.detection_model.eval()

        self.vision_mixer = nn.Linear(
                            conf.raw_vision_features_size + (0 if self.remove_vision_readout else conf.vision_object_emb_dim),
                            conf.vision_features_size
                        )

    def forward(self, img):
        # can run this on all images and cache on disk to save training/eval time

        raw_vision_features = self.vision_cnn(img)
        assert raw_vision_features.shape == \
            torch.Size([img.shape[0], self.raw_vision_features_size])

        if not self.remove_vision_readout:
            with torch.no_grad():
                self.detection_model.eval()
                detected_objs = self.detection_model.predict_classes(self.detection_model(img))
                vals, obj_idxs = detected_objs.topk(k=self.max_img_objects, dim=1)
                mask = (vals > 0)
                objects = obj_idxs * mask + self.object_na_idx * (~mask)

            objects_readout = self.object_embed_and_readout.forward(objects.int())

            out = self.vision_mixer(
                    torch.cat((objects_readout, raw_vision_features), dim=1)
                )
        else:
            out = self.vision_mixer(raw_vision_features)

        return out

    def get_img_transforms(self):
        return MultiLabelClassifier.get_img_transforms()
