import torch
import torch.nn as nn

import torch.nn.functional as F


def extract_curved_roi_features(features, center_points, boundary_points, aggregation='concat'):
    assert aggregation in ['sum', 'concat']

    # poi_feats = [[] for _ in range(len(features))]
    bs = features[0].shape[0]
    poi_feats = [[] for _ in range(bs)]



    for lvl, feats in enumerate(features):

        for batch_id, (feat, ref_pts) in enumerate(zip(feats, center_points)):
            sampled_features = F.grid_sample(feat[None,:,:,:], ref_pts[None,:,:,:], align_corners=True)
            # level_roi_feats.append(sampled_features)

            poi_feats[batch_id].append(sampled_features)

    for batch_id, feats in enumerate(poi_feats):
        if aggregation == 'sum':
            poi_feats[batch_id] = torch.stack(feats, dim=0).sum(dim=0).squeeze(0).permute(1,0,2) # [num_proposal, C, points]
        elif aggregation == 'concat':
            poi_feats[batch_id] = torch.cat(feats, dim=1).squeeze(0).permute(1,0,2)
        else:
            raise NotImplementedError

    return poi_feats


class CurvedRoIExtractor(nn.Module):
    """

    """
    def __init__(self, out_channels, out_height=None, sample_center=True, aggregation='sum', mode='align'):
        super(CurvedRoIExtractor, self).__init__()
        self.out_channels = out_channels
        self.mode = mode
        self.sample_center = sample_center
        if out_height is None:
            self.out_height = 3
        else:
            self.out_height = out_height

        if sample_center:
            assert self.out_height % 2 == 1

        assert aggregation in ['sum', 'concat']
        self.aggregation = aggregation


    def forward(self, features, center_points, boundary_points):

        bs = features[0].shape[0]

        if self.sample_center:
            assert center_points is not None

        if center_points is not None:
            assert len(boundary_points) == len(center_points)


        roi_feats_list = [[] for _ in range(bs)]
        t = torch.linspace(0, 1, self.out_height).to(features[0].device)
        t = t.reshape(self.out_height, 1, 1)
        t = t[None, :, :, :]

        for lvl, feats in enumerate(features):

            for batch_id in range(len(boundary_points)):

                # merge sampled coords
                up_points = boundary_points[batch_id][:, :, :2]
                down_points = boundary_points[batch_id][:, :, 2:]

                # for instance_id in range(boundary_points.size(0)):

                upts = up_points[:, None,:,:].repeat(1, self.out_height, 1, 1)
                dpts = down_points[:, None,:,:].repeat(1, self.out_height, 1, 1)
                sample_points = upts + (dpts - upts) * t.repeat(upts.size(0), 1, 1, 1)

                if self.sample_center:
                    sample_points = sample_points.transpose(0, 1)
                    sample_points[self.out_height // 2] = center_points[batch_id]
                    sample_points = sample_points.transpose(0, 1)

                encoded_feats = feats[batch_id]

                sampled_feats = F.grid_sample(encoded_feats[None, :, :, :].repeat(sample_points.size(0), 1, 1, 1),
                                              sample_points, align_corners=True)
                roi_feats_list[batch_id].append(sampled_feats)

        for batch_id, feats in enumerate(roi_feats_list):
            if self.aggregation == 'sum':
                roi_feats_list[batch_id] = torch.stack(feats, dim=0).sum(dim=0)  # [num_proposal, channel , height, points]
            elif self.aggregation == 'concat':
                roi_feats_list[batch_id] = torch.cat(feats, dim=1)
            else:
                raise NotImplementedError

        return roi_feats_list


